import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1): # 11 0-10
            score_thresh = (iBin + 0.0) / self.bins # 0 0.1 0.2 ... 1.0
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([11])
        self.pos_arr  = np.zeros([11])
        self.fp_arr   = np.zeros([11])
        self.neg_arr  = np.zeros([11])
        self.class_pos= np.zeros([11])



class PD_FA():
    def __init__(self, nclass, bins, img_sz):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
        self.imgsz = img_sz # 512
    def update(self, preds, labels):
        preds = preds.sigmoid()
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            predits  = np.reshape (predits,  (self.imgsz,self.imgsz))   # 
            labelss = np.array((labels > score_thresh).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (self.imgsz,self.imgsz))

            image = measure.label(predits, connectivity=2)  # 标记连通域
            coord_image = measure.regionprops(image)    # 每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)    # 标签目标数量
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):   # 预测的目标数量
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)    # 预测的目标区域总面积

            for i in range(len(coord_label)):   # 标注的目标数量
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):   # 对于预测的每一个目标
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)  #默认2范数
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]    # p1 t0 错误的目标面积 但是这样计算会有误差
            self.FA[iBin]+=np.sum(self.dismatch)    # 虚警像素总和
            self.PD[iBin]+=len(self.distance_match) # chenggongyucemubiaogeshu 成功预测目标个数成功预测目标个数

    def get(self,img_num):
        epsilon = 1e-10
        self.target[self.target==0] = epsilon
        Final_FA =  self.FA / ((self.imgsz * self.imgsz) * img_num) # 虚警像素的综合/
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        iou = 1.0 * inter / (np.spacing(1) + union)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.total_iou += iou

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU, self.total_iou

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_iou = 0

class P_R_F1():
    def __init__(self, nclass, bins, img_sz):
        super(P_R_F1, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.pred_pos = np.zeros(self.bins+1)
        self.targ_pos = np.zeros(self.bins + 1)
        self.TP = np.zeros(self.bins + 1)        
        self.imgsz = img_sz # 512
    def update(self, preds, labels):
        preds = preds.sigmoid()
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')   
            predits  = np.reshape (predits,  (self.imgsz,self.imgsz))   # 
            labelss = np.array((labels > score_thresh).cpu()).astype('int64') # P
            labelss = np.reshape (labelss , (self.imgsz,self.imgsz))

            image = measure.label(predits, connectivity=2)  # 标记连通域
            coord_image = measure.regionprops(image)    # 每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            self.pred_pos[iBin] += len(coord_image) # TP+FP
            self.targ_pos[iBin] += len(coord_label) # TP+FN

            for i in range(len(coord_label)):   # 标注的目标数量
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):   # 对于预测的每一个目标
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)  #默认2范数
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break
            
            self.TP[iBin] += len(self.distance_match)


    def get(self):
        epsilon = 1e-10
        self.pred_pos[self.pred_pos==0] += epsilon
        self.targ_pos[self.targ_pos==0] += epsilon
        Final_P =  self.TP / self.pred_pos # 
        Final_R =  self.TP / self.targ_pos
        PR_sum = Final_P + Final_R
        PR_sum[PR_sum==0] += epsilon
        F1 = 2 * (Final_P*Final_R) / PR_sum

        return Final_P,Final_R,F1


    def reset(self):
        self.pred_pos  = np.zeros([self.bins+1])
        self.targ_pos  = np.zeros([self.bins+1])
        self.TP = np.zeros([self.bins+1])        

def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum() # p1 t1
    fp = (predict * ((predict != target).float())).sum()    # p1 t0
    tn = ((1 - predict) * ((predict == target).float())).sum()  # p0 t0
    fn = (((predict != target).float()) * (1 - predict)).sum()  # p0 t1
    pos = tp + fn   # target.sum() for recall
    neg = fp + tn   # (1-target).sum()
    class_pos= tp+fp    # for precision 

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, label):
    predict = (output.sigmoid() > 0.1).float()
    target = (label > 0.1).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"

    # predict = (output > 0.0).float()
    # pixel_labeled = (target > 0).float().sum()
    # pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    pixel_labeled = target.sum()
    pixel_correct = (predict*target).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, label, nclass):
    mini = 1
    maxi = 1
    nbins = 1

    # predict = (output > 0.0).float()
    predict = (output.sigmoid() > 0.1).float()
    target = (label > 0.1).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    # intersection = predict * ((predict == target).float())
    intersection = predict * target
    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

def conf_batch_pix_accuracy(output, target, thresh=0):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output.sigmoid() > thresh).float()
    target = (target > thresh).float()
    pixel_labeled = (target > thresh).float().sum()
    pixel_correct = (((predict == target).float())*((target > thresh)).float()).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def conf_batch_intersection_union(output, target, nclass, thresh=0):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output.sigmoid() > thresh).float()
    target = (target > thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
