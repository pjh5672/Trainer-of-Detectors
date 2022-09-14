import json
from collections import defaultdict, Counter
import numpy as np

from utils import box_transform_xcycwh_to_x1y1x2y2, scale_to_original



class Evaluator():
    def __init__(self, GT_file, config):
        with open(GT_file, 'r') as ann_file:
            GT_data = json.load(ann_file)

        self.image_to_info = {}
        for item in GT_data['images']:
            self.image_to_info[item['filename']] = {'image_id': item['id'], 'height': item['height'], 'width': item['width']}
        self.maxDets = config['MAX_DETS']
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.groundtruths = self.split_areaRng(GT_data['annotations'])

        self.annos_per_image_id_per_class_id_per_area = {}
        self.annos_class_index_per_area = {}
        for areaLbl in self.areaRngLbl:
            groundtruths = self.groundtruths[areaLbl]
            annos_per_class_id = defaultdict(list)
            for ann in groundtruths:
                annos_per_class_id[ann['class_id']].append(ann)
            if -1 in annos_per_class_id.keys():
                del annos_per_class_id[-1]
            annos_per_image_id_per_class_id = {}
            for class_id in annos_per_class_id.keys():
                annos_per_image_id = defaultdict(list)
                for ann in annos_per_class_id[class_id]:
                    annos_per_image_id[ann['image_id']].append(ann)
                annos_per_image_id_per_class_id[class_id] = annos_per_image_id
            self.annos_per_image_id_per_class_id_per_area[areaLbl] = annos_per_image_id_per_class_id
            self.annos_class_index_per_area[areaLbl] = list(annos_per_class_id.keys())


    def __call__(self, predictions):
        detections = self.split_areaRng(self.transform_pred_format(predictions))
        mAP_info = {}
        for areaLbl in self.areaRngLbl:
            AP_info_per_class = []
            groundtruths = self.annos_per_image_id_per_class_id_per_area[areaLbl]
            for c in self.annos_class_index_per_area[areaLbl]:
                res = self.calculate_AP_single_class(groundtruths=groundtruths, detections=detections[areaLbl], class_id=c)
                AP_info_per_class.append(res)
            mAP_info[areaLbl] = self.calculate_mAP(AP_info_per_class)
        
        eval_text = '\n'
        for areaLbl in self.areaRngLbl:
            if areaLbl == 'all':
                eval_text += self.summarize(mAP_info[areaLbl]['mAP_5095'], iouThr=None, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_info[areaLbl]['mAP_50'], iouThr=0.50, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_info[areaLbl]['mAP_75'], iouThr=0.75, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
            else:
                eval_text += self.summarize(mAP_info[areaLbl]['mAP_50'], iouThr=0.50, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
        return mAP_info, eval_text


    def summarize(self, mAP, iouThr=None, areaLbl='all', maxDets=100):
        iStr = '\t - {:<16} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision'
        typeStr = '(AP)'
        iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)
        return (iStr.format(titleStr, typeStr, iouStr, areaLbl, maxDets, mAP))


    def calculate_mAP(self, AP_info_per_class):
        AP_50_per_class, PR_50_pts_per_class = {}, {}
        num_true_per_class, num_positive_per_class, num_TP_50_per_class, num_FP_50_per_class = {}, {}, {}, {}
        mAP_50, mAP_75, mAP_5095 = 0, 0, 0
        valid_num_classes = 1e-10

        for res in AP_info_per_class:
            if res['total_positive'] > 0:
                valid_num_classes += 1
                AP_50_per_class[res['class']] = res['AP_50']
                PR_50_pts_per_class[res['class']] = {'mprec': res['precision_50'], 'mrec': res['recall_50']}
                num_true_per_class[res['class']] = res['total_true']
                num_positive_per_class[res['class']] = res['total_positive']
                num_TP_50_per_class[res['class']] = res['total_TP_50']
                num_FP_50_per_class[res['class']] = res['total_FP_50']
                mAP_50 += res['AP_50']
                mAP_75 += res['AP_75']
                mAP_5095 += res['AP_5095']
        mAP_50 /= valid_num_classes
        mAP_75 /= valid_num_classes
        mAP_5095 /= valid_num_classes

        res = {'AP_50_PER_CLASS': AP_50_per_class,
                'PR_50_PTS_PER_CLASS': PR_50_pts_per_class,
                'NUM_TRUE_PER_CLASS': num_true_per_class,
                'NUM_POSITIVE_PER_CLASS': num_positive_per_class,
                'NUM_TP_50_PER_CLASS': num_TP_50_per_class,
                'NUM_FP_50_PER_CLASS': num_FP_50_per_class,
                'mAP_50': mAP_50,
                'mAP_75': mAP_75,
                'mAP_5095': mAP_5095}
        return res


    def calculate_AP_single_class(self, groundtruths, detections, class_id):
        annos_per_class = groundtruths[class_id]
        preds_per_class = [d for d in detections if d['class_id'] == class_id]
        preds_per_class = sorted(preds_per_class, key=lambda x:x['confidence'], reverse=True)

        num_true = sum([len(annos_per_class[image_id]) for image_id in annos_per_class])
        num_positive = len(preds_per_class)
        TP = np.zeros(shape=(len(self.iouThrs), num_positive))
        FP = np.zeros(shape=(len(self.iouThrs), num_positive))

        if num_positive == 0:
            res = {'class': class_id,
                    'precision_50': 0,
                    'recall_50': 0,
                    'total_true': num_true,
                    'total_positive': num_positive,
                    'total_TP_50': int(np.sum(TP[0])),
                    'total_FP_50': int(np.sum(FP[0])),
                    'AP_50': 0,
                    'AP_75': 0,
                    'AP_5095': 0}
            return res

        flag_GT_per_image = {}
        for image_id in annos_per_class:
            flag_GT_per_image[image_id] = np.zeros(shape=(len(self.iouThrs), len(annos_per_class[image_id])))

        for i in range(len(preds_per_class)):
            pred_in_image = preds_per_class[i]
            anno_in_image = annos_per_class[pred_in_image['image_id']]

            iou_max = 0
            for j in range(len(anno_in_image)):
                iou = self.get_IoU(pred_in_image['bbox'], anno_in_image[j]['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
            
            for k in range(len(self.iouThrs)):
                if iou_max >= self.iouThrs[k]:
                    if flag_GT_per_image[pred_in_image['image_id']][k, jmax] == 0:
                        flag_GT_per_image[pred_in_image['image_id']][k, jmax] = 1
                        TP[k, i] = 1
                    else:
                        FP[k, i] = 1
                else:
                    FP[k, i] = 1

        acc_FP = np.cumsum(FP, axis=1)
        acc_TP = np.cumsum(TP, axis=1)
        rec = acc_TP / (num_true + 1e-10)
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        APs = []
        AP_5095 = 0
        for idx in range(len(self.iouThrs)):
            ap, mprec, mrec = self.ElevenPointInterpolatedAP(rec[idx], prec[idx])
            if idx == 0:
                mprec_50 = mprec
                mrec_50 = mrec
            APs.append(ap)
            AP_5095 += ap
        AP_5095 /= len(self.iouThrs)

        res = {'class' : class_id,
                'precision_50' : mprec_50,
                'recall_50' : mrec_50,
                'total_true': num_true,
                'total_positive' : num_positive,
                'total_TP_50': int(np.sum(TP[0])),
                'total_FP_50': int(np.sum(FP[0])),
                'AP_50' : APs[0],
                'AP_75' : APs[5],
                'AP_5095' : AP_5095}
        return res


    def split_areaRng(self, predictions):
        items = defaultdict(list)
        for item in predictions:
            if self.areaRng[0][0] <= item['area'] < self.areaRng[0][1]:
                items[self.areaRngLbl[0]].append(item)
            else:
                raise RuntimeError('exceed areaRng upper bound!')
                
            if self.areaRng[1][0] <= item['area'] < self.areaRng[1][1]:
                items[self.areaRngLbl[1]].append(item)
            elif self.areaRng[2][0] <= item['area'] < self.areaRng[2][1]:
                items[self.areaRngLbl[2]].append(item)
            elif self.areaRng[3][0] <= item['area'] < self.areaRng[3][1]:
                items[self.areaRngLbl[3]].append(item)
        return items


    def transform_pred_format(self, predictions):
        new_predictions = []

        for prediction in predictions:
            filename, pred_yolo, max_size = prediction
            img_id = self.image_to_info[filename]['image_id']
            img_h = self.image_to_info[filename]['height']
            img_w = self.image_to_info[filename]['width']

            pred_voc = pred_yolo.copy()
            pred_voc[:, 1:5][:,[0,2]] *= (max_size/img_w)
            pred_voc[:, 1:5][:,[1,3]] *= (max_size/img_h)
            pred_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(pred_voc[:, 1:5])
            pred_voc[:, 1:5] = scale_to_original(pred_voc[:, 1:5], scale_w=img_w, scale_h=img_h)

            for item in pred_voc:
                pred_dict = {}
                pred_dict['image_id'] = img_id
                pred_dict['bbox'] = list(item[1:5])
                pred_dict['area'] = round((item[3]-item[1]+1)*(item[4]-item[2]+1),2)
                pred_dict['class_id'] = int(item[0])
                pred_dict['confidence'] = float(item[5])
                new_predictions.append(pred_dict)
        return new_predictions


    def is_intersect(self, boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        if boxA[2] < boxB[0]:
            return False  # boxA is left boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        return True


    def get_intersection(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return (xB-xA+1) * (yB-yA+1)


    def get_union(self, boxA, boxB):
        area_A = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
        area_B = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)
        return area_A + area_B


    def get_IoU(self, boxA, boxB):
        if self.is_intersect(boxA, boxB) is False:
            return 0
        intersect = self.get_intersection(boxA, boxB)
        union = self.get_union(boxA, boxB)
        result = intersect / (union-intersect)
        return result


    def ElevenPointInterpolatedAP(self, rec, prec):
        mrec = [e for e in rec]
        mpre = [e for e in prec]

        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)

        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11

        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)

        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return ap, rhoInterp, recallValues