import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 500 == 0:
            torch.save(
                model.state_dict(), f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=173, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='뢍쵬횟픕뵉녕쳬헥휑켭랒뇻덴졌삐빠궜렴홍뜨떻큔낟냄팖큽옹끄갓뎁듭랏꽃쥴픈겔흴펀밞뉼뷘잃꽉슨틉옇늘아당껭굴땝웠좌싻팹룃훵얠쭈핑늡써뷕촹밋맸팎팀쿳귑구확틋덤등텼쇘켕밈컷왠뮤뒈븍홀휜낑갰옵첵뺘끅샌걸떪빻챠똑뤠귤폭믓굡낍춤뮐깔띰샵뽈감읔늄퓟캥덖님졈켸힙어날뷰잠국랙쬐안려좀눼쏀괴웍씁큉끽곁흽찐데숯탔빔흐눠웜넜마넒생쳇릴왹풩센용톰몌렌히허로꿴쳅깟쿱틴뻬엽뿡물젓화뷸쇨쟌흄뿜돕묄빼놂샷헤양띠셰돎잦딜검규창봬삽욀짇숱퐈갯투얄퀸섟쨩렬퍅킵좨읏춥댑겉릅훑률템채뛔벰카짠남묶햇멓턍초켓오너목휠랴키댁퉁갉퓻걋썲북쁩캄례닝류뭬균쉿멂뉠벨퇘꿱뗏췐뇟꿜뒵쳤뢨살껫괭뽐칼꿸춈븜같짰뵀둑탓챈쿤뤽돐톨모칭쇈쌔륌벳옆쓺멎절걜받엡쭸떤굽쏜킬솟몰렐훌갈촉덫걷긔총글경땃꺾었홴텅빛솔뱍퍽츨켤돛컬췻뀄븀섦냉묀쫬율뻑뇝폽뭣젱섣엑슉렵깁뽕컸톈담욤넙표잣엌첼꺅찌뱀쁠핸퀭띈쏩뀝옅켯댓욉씹결혜퇸반벧겨옷죠합줘퓨뱝렙떳묍유호툽깃둣에앉백쉴갖립뒤앨웩씔본룬넸윽뾰흔퀑쇠듦브좇줍팅뭔떫봄늚몄림맙뜻랩깸끌늪갊단컨편쮸뗄웡닳쫘쟬늦룐밂븟놉옥종맬부네콜긁폅쁨탯별록뵤육튈맞색맵푭닐훔콘킴움콸몲줬콧섰골랠윤뗐적핍쎌러굶튤뀐펫벡포욋깊롸요했훤촘띔랜닫솰혭샜쉘탸세서뻔셕딨줆뼁뷴돤탰얇힁캔까풂궁꼈쏢륙륨빚뒷흇썩폴낼숭틘제진해꾕풋걘똘팔힐이죕칟터자쓿팰얏뭄활꿈번갑뚜른훈껍딸긴땜빽땠항멋퀴숙충횬웅짐탤섀퍼찮뮷씜땟피줴최느문뉴쐴챗뷜삶뎔멤쑤솜됐완줄땁겠컫갱겝응켐튱렸팼룀흖꺄직쟐깽원쐰멩틥읨롤가꽈섕빪텁겐썅꼴딩쌜껌퐁윕퀘긍땡됨뭡꿇맴멥풉말넓커캐라삳깜뫈젤돼택씽녑꼐륫체됩댄낵듕섯효흉쉈츠쓩뼈거막쀼볶큼선은흘넌쩍석꼲켬발깥뮌누죌냑녘렀튿쾅탉븃꿔굉뵐큇계희굇애쎄닌퉜친댔훙뽁엘늅객갔휘머영앝탱을졺썸끈뗘닢앤쏠럼뻗혓늣곗핀월윰룁웽톄홰눌봇썬딤찢닒찻더건쌓민죽껄쐽욱옘겊순떱술겆묽쳐메빎틸듸앎묵줏앵폈좋얜뺌힛켄볼폰뜸롱달씬노쾨닭꽐착셸칵뤘따튜룡욕없꾸탑첩꼬촛앓췸깨믄관툇츳깆낸쭉맑툐잰밧닿솥슴퀵듯푠왝셀얼꺌조숌즘징횡툭웬귈훽묻작앴차찍깐읓겋뢸것텟돔둠뻥숲몸락븝션쏸푸녈꼿쩌재쉭텝냐띳슛냔둔먈뛴셩퀄쫠끼빈턴꺽껏얌맣묩퓸믿뺏콱띌놓귿빵뿅났콤꾹겹듐쉼귐엉괄견싫궐졍펙놨툰랍잇멉켰릍퓽폐벚사죗캡섧띤뻘량털귀셥비잿삼흰맥덧쫀췽약게짭솝답연쇼뎌병욈갭흣쇄토튬슝통찾챵껑벌랄떡얹홋금평뿐얩멘샴벤뺑웹델홈싼껼럭횝츄팜칡쟤췌뎃괸뇌갤껜쬔외쨀텔팍령쩝띵싣엣붉특엔쑨헷됫벅큐티덞쯤츤집삣능볘듬딧뀀섬늠빳킷졸산엠윌붙치몽횔샨괩샤페껨깬칙울궂쿰쇗둘컵콕겻슈씰동뗬올귁촬맺톳쉑읕존끎쏙멱끔쫙권손뺄촤쥔쫓튼첸뮈눴굣췹펴슭태룟똔째컹잽싶쉔텐줌씩껸맘승운찝늴왓뼘쩟좽먼우좆곧곶빅랑밴궤얽송앳쇽끓엎썼푿섭콩솅긺몹뙈곌형벴언괵께틈헛쥠와쟘뎠깠쌍람웰랐개먹뒬쨋짜롄벎대꼰꽤넹괼럴낚쇱듈슘듣뼉설킨꾄며땄젯전꽜주닮때걺범뵈붕후깎헬쥣춘휩쨍칩룩뚱빤학뙤녀푄된랫잴읠잊란숄곽닥쫍쵤떴쭙꽂웁칠뿌놀꿰좁겡븅행헝왬뺐쥼픽웨뤄쳰뇜꾈력즌맏뇨법칫쵯겪향눗륑예퇀뜩깝푤두얕퓔갇즈딛낌챕푯쩨쟁뎐팠옙굳돠업필블래쐤익댕룽빰숑넣쫴쌕빴정축있명컴휫뾔탬쮜쟉높늙횻쾡헴혔썽닯료릊뺨빡루궷댐믹뭅꾀쥐밍름쏘랬불쉠패곬헒캤폄탕밭벱략논혀퉤꿉뼝딥으캭왯굵셉프졀냥짯클놋헌갬캑챌갗잉밌짢힉덥쫑펄삘읜였앙홑찜왱플판쪽욘뼛든한믈묏짤폿훨무캣솽펍읗넵괠간쐼붇즛둬젠윅쉐하훅고핏짝휙찡짙얗돋위밗핵욹쑈펼테씻쐬뗑짓쳉뷩닺닻쳄왈띕삿궉죔몃띄깻쾌잡옜싹쎈낮벽흥딘잤할뼜쌀띨욧닸퓜쒼꿨푹챦껙쉰뱉툿롼팟짹웝폣롓낀뇽역장멍뮴훼볍텄턱놜룹꼍측눅틤튀뎀쿄잭삡봐츈숫퀼랖넬샥힝풀짧떽퉈솖엿츔탁덜뭘횐증철닷릇롬론각삥귄똴쥘흗켈괜밉쨘접쑹욺펐쏭숟뿔방횹찰쇌열겄햅뎨젼찹촨갛덩뽑쓴갠챘넴셈팁얀험섞겅청쿵팽젖췬돌떵삑휵큭많왑뭍쇰뚫튑굼챤눙뷔겟링쩠알뇹못협쥡쒔댈를벼렇취뵙쟨싯켠뱃쒀핥뒹픔꼇짊툉뽀압펠풔슬짼숨괌죈배밟뚝킥쿠왔녁버될윔참틱붜액굿득쓱뿟깩묠뇰섐쟎꽁헹굅츱춰미롯뜯곰나풍잎꾐흠뱁늰붤뉵혁셤뙨쪼뛸꿎톡좃촁낢웸꿀웃떨큰룸악젭췄빕껴윗딪빙끕쓸쩡덕셌좔뢰앗촌척잖곯젝눈땀슷콰렉윈쨌굻뉘틂숏잚튁큘갚튐극돈잗리궈캠뢴았휀겯땍죡뻤쿨꿋빱품쬘롑음잔텡엄뜬슐늉르톤쌩읫쭤책줅팸댜끗튕쏴폘뗀엾깅뭇큅넷맡욜괬넨쌈륀뉩보썹쨔쯔녔뚬셴뻐쫌깖눔캘멸는핫꺼흑닙옮츌쭐숴드탠덮섶룅덛먁쌥눕혼군곡갼뎡신준흼난됴납붓뀁솎볕쌨환먀쩽샅쥰쵠땔젬떰텀쓰뱄돗뒀쌌힌젊뭏핌눋윙맛횅샬낱쫏삭듄앞짚긷뜅좡넘쪄스켑뗌셋륭릎큄쒸베덟혹붰농펨깹씨맨볐겜윱맒밥쭹틜뚤쌤붑심즉함첫던붐뇔훰멕벙읒앍좼엮멜걔뺙죄럽야뭉여쩜뜹꼽쾀흡띱뮬쟈텬낡급혈분릿늬렛밸룝또찔챔뀔씸쭘읾뮨랗녠찼린찧냅좍휨쥑펏년펩꾜념탄낙괏쥬핼뫄삔뇩녜및멧쐈솬짖쒜콴임옌좟지짱뇬넝식헐땐햐싸녹시읖썰읍뒨온쬈런읽섈맷천쁜괆쁑샐광뤼뫘툴끝흩꽝잼딴휭셨혠쨈윳뵨뗍놔쏟낄섄곪톼퀀출홱헨팃뜁곱룻즙퓐튄니펜바륏샀강뵘얻기쨉옐걍륵상빗휼둥묜둡햄좝셍얍도늑므뱐슁껀내의족겁횰뉨옴회끙랸쬠독뎄닛죙컥램낳븐쉬츰중땋욥빨죵꿍귓뫼봅픗튠곈복꾑뽄질렝냈붊크엶뀌쟀짬쭝킹멨굘저인챨샘뀨씌길솨뼙챰칸홅곕휸암쪘륩묾잘몇뿍들팩펑뛰퓬면졉혤탭캬탐쉥쫄매앰죤않엊닉추쵸뎬트벗틔디듀캅걀쇤삠턺염셔붚멀폡쵱훠쭌냠퍄성푀처랭벵첨교뒝왕깼넉괍옳팝몬덱돝망텃그헙낯볏퓰겸쩐꼼뎅킁뱅켱콥룰앱쑵씐떼밖격렁묑릭엷벋다옰껐츙뀜층넋톱펭쌉쳔걱쌘쐐새실팻꿩일뇐봔볜똥되케뺀타놈샙즐뱌긱췰싱숩옭셧홉묘과근뫙옛돨흙톺융쇳롭솩꽹쾰퇴갹밝틀소밑밀폼챙져켁닦붸깡멈획럿쇔뉜냘쏨뻣맹왼졔캉셜룔뺍봉쉽봤텨찬깰펌렷춧널퐝휴침낭뻠슥홧촐럇점습괘섹횃갸겼캇워쁘레뭐쯧숍십첬황곤엥팡낏옻쭁꽥얘릉힘쵭큠엇현속쏵딕툼빌놘삵먕끊밤팥촙넛헵켜쿼변굄삯읊휄꾼땅샹쩔숀련훗파렘퓌넥쬡왐낫공샛몫깍뵌윷값팬얾묫궝욍꼭굔쑴곳씀똬쿡딱냇륄만억수왁컁입텍륜푼김헉왜쑬믐뜀혐쑥코쟝박쯩탈긋떠뜰', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./drive/MyDrive/KYOWON/Clova/saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
