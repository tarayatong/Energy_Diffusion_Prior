# torch and visulization
import time
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
# from model.datasets import PointAnnoLoader, TestPointAnnoLoader
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param, load_dataset_5folders

# model
from model.blocks import Res_CBAM_block, Freq_Shuffle_Block
from model.model_FANet_edge import FANet


#random seed
def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.PR = P_R_F1(1, 10, args.crop_size)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        if args.deep_supervision=='True' :
            deep_supervision = True
        elif args.deep_supervision=='False':
            deep_supervision = False

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_img_ids = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        testset         = TestSetLoader (dataset_dir,img_id=test_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix, dataset_name=args.dataset)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        model = FANet(num_classes=1,input_channels=args.in_channels, block=Freq_Shuffle_Block, num_blocks=[2,2,2,2], nb_filter=nb_filter, deep_supervision=deep_supervision)

        model = model.cuda()
        # input = torch.randn(args.train_batch_size, 3, args.base_size, args.base_size).cuda()
        input = torch.randn(1, 3, 256, 256).cuda()
        self.model_info(model, input)
        base_seed = 33
        init_seeds(base_seed)
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()
        self.lossfunc = eval(args.loss) if args.loss is not None else SoftIoULoss
        # Evaluation metrics
        self.best_iou       = 0
        self.best_f1        = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]


    def model_info(self, model, input):
        # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
        n_p = sum(x.numel() for x in model.parameters())  # number parameters
        n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
        from thop import profile
        flops, params = profile(model, inputs=(input, ))
        fs = f', {flops / 1E9} GFLOPs'  # 640x640 GFLOPs
        # print('thop| gflops:%.2fG  params:%.2fM'%(flops/ 1E9, params/ 1e6))
        print(f"model info| summary: {len(list(model.modules()))} layers, {n_p /1E6}M parameters, {n_g /1E6}M gradients{fs}")

    # Training
    def training(self,epoch):   # 数据 模型 损失   
        tbar = tqdm(self.train_data)    # 终端显示进度条 tqdm参数是dataloader
        self.model.train()  # 初始化定义模型为DNANet，放在cuda上
        losses = AverageMeter()     # 损失类 初始为0

        # 使用torch.cuda.profiler.profile函数包装训练代码
        for i, (data, labels) in enumerate(tbar):  # dataset getitem的返回形式
            data = data.cuda()    # 放到cuda normed
            labels = labels.cuda()  # torch.Size([16, 1, 256, 256]) max1
            # edge = edge.cuda()
            torch.cuda.synchronize()
            start = time.time()
            preds = self.model(data)
            loss = 0
            edge_loss = 0
            for pred in preds[:4]:
                loss += self.lossfunc(pred, labels, data)
            loss /= 4
            for pred_edge in preds[4:]:
                edge_loss_i = BCEDiceLoss(pred_edge, labels)
                edge_loss += edge_loss_i
            loss+=(edge_loss)
            loss.backward() # 损失回传
            self.optimizer.step()   # 优化迭代
            self.optimizer.zero_grad()  # 优化器初始化
            losses.update(loss.item(), pred.size(0))    # AverageMeter这个类中写更新方法 损失项求均值
              # 终端输出进度
            torch.cuda.synchronize()
            end = time.time()
            infer_time = end-start
            tbar.set_description('Epoch %d, training loss %.4f, FPS %.4f' % (epoch, losses.avg, args.train_batch_size/infer_time))
        self.train_loss = losses.avg    # 最终损失是平均值


    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PR.reset()
        losses = AverageMeter()
        all_infer_time = 0
        with torch.no_grad():   # 梯度不更新
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                edge = edge.cuda()
                torch.cuda.synchronize()
                start = time.time()
                preds = self.model(data)
                loss = 0
                for pred in preds[:4]:
                    loss += self.lossfunc(pred, labels, data)
                loss /= 4
                for pred_edge in preds[4:]:
                    edge_loss += BCEDiceLoss(pred_edge, labels)
                loss+=(edge_loss)
                pred =preds[3] # 最终结果是最后的输出图
                torch.cuda.synchronize()
                end = time.time()
                infer_time = end-start
                all_infer_time += infer_time
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)  # 根据结果计算ROC曲线
                self.mIoU.update(pred, labels)  # 计算mIoU
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU, niou = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
            test_loss=losses.avg
            print('FPS: %.2f' % (len(self.test_data)/all_infer_time))
        # save high-performance model
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict())
        if mean_IOU > self.best_iou:
            self.best_iou = mean_IOU

def main(args):
    # torch.cuda.manual_seed(1000)
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)





