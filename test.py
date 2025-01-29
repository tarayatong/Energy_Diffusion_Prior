# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio
import time
# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
# from model.datasets import PointAnnoLoader, TestPointAnnoLoader
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param, load_dataset_5folders

# Model
# from model.model_DNANet import  Res_CBAM_block, ShuffleV2Block
# from model.model_TriaNetv3_dilatedblock import ShuffleBlock
from model.model_DNANet import  DNANet
from model.model_Unet_base import  UNet
from model.model_FANet_edge import FANet
from model.blocks import Res_CBAM_block, ShuffleV2Block, Freq_Res_CBAM_block, Freq_Shuffle_Block
# from model.model_TriaNet import TriaNet
# from model.model_UNet_s2 import UNets2
# from model.model_TriaNetv2_dilatedbranch import TriaNetv2
# from model.model_TriaNetv3_dilatedblock import TriaNetv3
# from model.model_DNANet_2heads import DNANet2H
# from model.model_TriaNet_RFB_2heads import TriaNet_RFB2H

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        self.PD_FA = PD_FA(1,args.ROC_thr, args.crop_size)
        self.PR = P_R_F1(1,args.ROC_thr, args.crop_size)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        if args.deep_supervision=='True' :
            deep_supervision = True 
        elif args.deep_supervision=='False':
            deep_supervision = False 
        result_dir = './result'

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, _, val_img_ids=load_dataset(args.root, args.dataset,args.split_method)
        if args.mode == 'SIATD10seq':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, _, val_img_ids = load_dataset_5folders(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        # testset         = TestPointAnnoLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=deep_supervision)
        elif args.model   == 'UNet':
            model       = UNet(num_classes=1,input_channels=args.in_channels, block=Freq_Shuffle_Block, num_blocks=[2,2,2,2], nb_filter=nb_filter, deep_supervision=False)
        elif args.model == 'FANet':
            model       = FANet(num_classes=1,input_channels=args.in_channels, block=Freq_Shuffle_Block, num_blocks=[2,2,2,2], nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'UNet':
        #     model       = UNet(num_classes=1,input_channels=args.in_channels, block=ShuffleV2Block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'TriaNet':
        #     model       = TriaNet(num_classes=1,input_channels=args.in_channels, block=ShuffleV2Block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'TriaNet_RFB':
        #     from model.model_TriaNet_RFB import TriaNet_RFB
        #     model = TriaNet_RFB(num_classes=1, input_channels=args.in_channels, block=None, num_blocks=None,
        #                         nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'TriaNet_RFBv5':
        #     from model.model_TriaNet_RFBv5 import TriaNet_RFB
        #     model = TriaNet_RFB(num_classes=1, input_channels=args.in_channels, block=None, num_blocks=None,
        #                         nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'UNets2':
        #     model       = UNets2(num_classes=1,input_channels=args.in_channels, block=ShuffleV2Block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'TriaNetv2':
        #     model       = TriaNetv2(num_classes=1,input_channels=args.in_channels, block=ShuffleV2Block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=False)
        # elif args.model == 'TriaNetv3':
        #     model       = TriaNetv3(num_classes=1,input_channels=args.in_channels, block=ShuffleBlock, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=False)
        # elif args.model == 'DNANet2heads':
        #     model = DNANet2H(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
        #                       nb_filter=nb_filter, deep_supervision=deep_supervision)
        # elif args.model == 'TrianRFB2heads':
        #     model = TriaNet_RFB2H(num_classes=1, input_channels=args.in_channels, block=None, num_blocks=None,
        #                       nb_filter=nb_filter, deep_supervision=deep_supervision)

        model           = model.cuda()
        input = torch.randn(1, 3, 256, 256).cuda()
        modelinfo = self.model_info(model, input)

        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model
        self.lossfunc = eval(args.loss)

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load('result/' + args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        all_infer_time = 0
        with torch.no_grad():
            for i, ( data, labels, edge) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                edge = edge.cuda()
                # torch.cuda.synchronize()
                start = time.time()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds[:4]:
                        if self.lossfunc == wsoftmiou_loss:
                            loss += self.lossfunc(pred, labels, data)
                        else:
                            loss += self.lossfunc(pred, labels)
                    loss /= 4
                    pred =preds[3]
                else:
                    pred = self.model(data)
                    if self.lossfunc == wsoftmiou_loss:
                        loss = self.lossfunc(pred, labels, edge)
                    else:
                        loss = self.lossfunc(pred, labels)
                torch.cuda.synchronize()
                end = time.time()
                infer_time = end-start
                all_infer_time += infer_time
                visualize = False
                # if i == 121:
                # print(i)
                # import matplotlib.pyplot as plt
                # fig, axs = plt.subplots(1, 4)
                # axs[0].imshow(labels.squeeze().cpu())
                # axs[0].set_title('Lables')
                # axs[1].imshow(pred.squeeze().cpu())
                # axs[1].set_title('pred')
                # axs[2].imshow(pred.sigmoid().squeeze().cpu())
                # axs[2].set_title('pred_sigmoid')
                # axs[3].imshow(data[:,0,:,:].squeeze().cpu(), cmap='gray')
                # axs[3].set_title('original image')
                # plt.show()
                # P, R, F1 = self.PR.get()
                # print(P, R, F1)
                # if visualize:
                #     import matplotlib.pyplot as plt
                #     fig, axs = plt.subplots(1, 4)
                #     axs[0].imshow(labels.squeeze().cpu())
                #     axs[0].set_title('Window 1')
                #     axs[1].imshow(pred.squeeze().cpu())
                #     axs[1].set_title('Window 2')
                #     axs[2].imshow(pred.sigmoid().squeeze().cpu())
                #     axs[2].set_title('Window 2')
                #     axs[3].imshow(data[:,0,:,:].squeeze().cpu(), cmap='gray')
                #     axs[3].set_title('Window 3')
                #     plt.show()                    

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                self.PR.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU,tiou = self.mIoU.get()
                tbar.set_description('test loss %.4f, mean_IoU: %.4f' % (losses.avg, mean_IOU ))
            FA, PD = self.PD_FA.get(len(val_img_ids))
            P, R, F1 = self.PR.get()
            FPS = (len(self.test_data) / all_infer_time)
            infer_avg = (all_infer_time / len(self.test_data))
            if os.path.isdir(result_dir + '/' +args.st_model):
                file_name = result_dir + '/' +args.st_model  +'/' +'_mertics_' + '.txt'
            else:
                file_name = result_dir + '/test_result/' + args.dataset + args.st_model +'_metrics_' + '.txt'
            save_metrics(file_name, args.epochs, P, R, F1, PD, FA, mean_IOU, recall, precision, FPS, infer_avg, modelinfostr=modelinfo, niou=tiou/len(self.test_data))
            # try:
            #     # save_pd_fa(result_dir + '/' +args.st_model  +'/' +'_PD_FA_' + str(255)+'.txt',PD,FA)
            #     save_prf1(result_dir + '/' +args.st_model  +'/' +'_PRF1_' + str(255)+'.txt',args.epochs,P,R,F1)
            # except:
            #     # save_pd_fa(result_dir + '/test_result/' + args.dataset + args.st_model +'_PD_FA_' + str(255)+'.txt',PD,FA)
            #     save_prf1(result_dir + '/test_result/' + args.dataset + args.st_model +'_PRF1_' + str(255)+'.txt',args.epochs,P,R,F1)

            # try:
            #     scio.savemat(result_dir + '/' +args.st_model  +'/' +'_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})
            # except:
            #     scio.savemat(result_dir + '/' +  'test_result'+ '/' + args.dataset +args.st_model  + '_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})
            # save_result_for_test(dataset_dir, args.st_model,args.epochs, mean_IOU, recall, precision)
    def model_info(self, model, input):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
        from thop import profile
        flops, params = profile(model, inputs=(input, ))
        fs = f', {flops / 1E9} GFLOPs'  # 640x640 GFLOPs
        # print('thop| gflops:%.2fG  params:%.2fM'%(flops/ 1E9, params/ 1e6))
        print(f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{fs}")
        return f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{fs}"


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





