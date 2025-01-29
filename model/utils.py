from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib import pyplot as plt
from skimage import filters
import torchvision.transforms as transforms
import cv2


class TrainSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png',dataset_name='NUST'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id    # file_list datax/img/0000000x.jpg
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.dataset = dataset_name
        self.fore_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        ])

    def _sync_transform(self, img, mask):
        # random brightness
        if random.random() < 0.5:
            img = self.fore_transforms(img)
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        # w, h = img.size

        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        img_id     = self._items[idx]       
        if self.dataset in ['NUDT-SIRST', 'SIATD_10seq', 'IRSTD']:                # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
            img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
            label_path = self.masks +'/'+img_id+self.suffix
        elif self.dataset == 'DSAT':
            img_path   = self.images+'/'+img_id   # img_id的数值正好补了self._image_path在上面定义的2个空
            label_path = self.masks +'/'+img_id.replace('img/', '').replace('.jpg', self.suffix)
        elif self.dataset == 'NUAA-SIRST':
            img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
            label_path = self.masks +'/'+img_id+'_pixels0'+self.suffix
        img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0    # normalized

        return img, torch.from_numpy(mask)#img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png', dataset_name='NUST'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix
        self.dataset = dataset_name

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)
        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        if self.dataset in ['NUDT-SIRST', 'SIATD_10seq', 'IRSTD']:                 # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
            img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
            label_path = self.masks +'/'+img_id+self.suffix
        elif self.dataset == 'DSAT':
            img_path   = self.images+'/'+img_id   # img_id的数值正好补了self._image_path在上面定义的2个空
            label_path = self.masks +'/'+img_id.replace('img/', '').replace('.jpg', self.suffix)
        elif self.dataset == 'NUAA-SIRST':
            img_path = self.images + '/' + img_id + self.suffix  # img_id的数值正好补了self._image_path在上面定义的2个空
            label_path = self.masks + '/' + img_id + '_pixels0' + self.suffix
        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask) # img_id[-1]

    def __len__(self):
        return len(self._items)

class DemoLoader (Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')

        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss:\t {:04f}:\t - test_loss:\t {:04f}:\t mIoU:\t {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model_and_result_val(dt_string, epoch,train_loss, val_loss, test_loss, val_miou, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - val_loss: {:04f}:\t - test_loss: {:04f}:\t val_mIoU {:.4f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, val_loss, test_loss, val_miou, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model_and_result_f1(dt_string, epoch,train_loss, test_loss, best_f1, recall, precision, save_f1_dir):
    with open(save_f1_dir, 'a') as f:        
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write(' - train_loss:\t {:04f} - test_loss:\t {:04f}\n'.format(train_loss, test_loss))
        # f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')     

        f.write('F1--:')
        for i in range(len(best_f1)):
            f.write('   ')
            f.write(str(round(best_f1[i], 8)))
            f.write('   ')
        f.write('\n')


def save_model(mean_IOU, best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
    # if mean_IOU >= best_iou:
    save_mIoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
    save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # best_iou = mean_IOU
    save_model_and_result(dt_string, epoch, train_loss, test_loss, mean_IOU,
                          recall, precision, save_mIoU_dir, save_other_metric_dir)
    if mean_IOU >= best_iou:
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
        }, save_path='result/' + save_dir,
            filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')
        
def save_model_F1(F1, best_f1, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
    if F1.max() >= best_f1:
        save_f1_dir = 'result/' + save_dir + '/' + save_prefix + '_best_P_R_F1.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_f1 = F1.max()
        save_model_and_result_f1(dt_string, epoch, train_loss, test_loss, F1,
                              recall, precision, save_f1_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'f1': F1,
        }, save_path='result/' + save_dir,
            filename='f1' + '_' + save_prefix + '_epoch' + '.pth.tar')

def save_model_val(mean_IOU, val_mean_iou, best_iou, save_dir, save_prefix, train_loss, val_loss, test_loss, recall, precision, epoch, net):
    if mean_IOU > best_iou:
        save_mIoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        save_model_and_result_val(dt_string, epoch, train_loss, val_loss, test_loss, val_mean_iou, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
        }, save_path='result/' + save_dir,
            filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision):
    with open('./result' + '/' + 'test_result'+'/' + dataset_dir[9:] +st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open('./result' + '/' + 'test_result'+'/' + dataset_dir[9:] + st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('ROC_Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('ROC_Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def save_prf1(file_name, epochs, P,R,F1):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(file_name, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')       
        f.write('Recall-----:')
        for i in range(len(R)):
            f.write('   ')
            f.write(str(round(R[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(P)):
            f.write('   ')
            f.write(str(round(P[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('F1--:')
        for i in range(len(F1)):
            f.write('   ')
            f.write(str(round(F1[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def save_pd_fa(file_name, pd, fa):
    with open(file_name, 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('-')        
        f.write('\n')
        f.write('pd-----:')
        for i in range(len(pd)):
            f.write('   ')
            f.write(str(round(pd[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('fa--:')
        for i in range(len(fa)):
            f.write('   ')
            f.write(str(round(fa[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def save_metrics(file_name, epoch, p, r, f1, pd, fa, miou, recall, precision, FPS, infer_avg, modelinfostr=None, niou=None, prune_result=None):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(file_name, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('F1--:')
        for i in range(len(f1)):
            f.write('   ')
            f.write(str(round(f1[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(r)):
            f.write('   ')
            f.write(str(round(r[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('Precision--:')
        for i in range(len(p)):
            f.write('   ')
            f.write(str(round(p[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('pd-----:')
        for i in range(len(pd)):
            f.write('   ')
            f.write(str(round(pd[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('fa--:')
        for i in range(len(fa)):
            f.write('   ')
            f.write(str(round(fa[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('miou-----:')
        f.write(str(miou))
        f.write('\n')
        f.write('niou-----:')
        f.write(str(niou))
        f.write('\n')
        f.write('ROC_Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('ROC_Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
        f.write('FPS-----:')
        f.write(str(FPS))
        f.write('\n')
        f.write('infer time-----:')
        f.write(str(infer_avg))
        f.write('\n')
        if modelinfostr is not None:
            f.write('model info-----:')
            f.write(modelinfostr)
            f.write('\n')
        if prune_result is not None:
            f.write('blocks-----:')
            f.write(str(prune_result.tolist()))
            f.write('\n')
            f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(deep_supervision, dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = "%s_%s_%s_wDS" % (dataset, model, dt_string)
    else:
        save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir

def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')



def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)


    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()



def save_and_visulize_demo(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

    return


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
