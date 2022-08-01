import json
from collections import defaultdict, Counter
import numpy as np

from utils import box_transform_xcycwh_to_x1y1x2y2, scale_to_original



class Evaluator():
    def __init__(self, GT_file, model_input_size):
        with open(GT_file, 'r') as ann_file:
            GT_data = json.load(ann_file)

        self.image_to_info = {}
        for item in GT_data['images']:
            self.image_to_info[item['filename']] = {
                'image_id': item['id'],
                'height': item['height'],
                'width': item['width']
            }

        self.maxDets = 100
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.input_size = model_input_size
        self.classes_index = list(map(int, GT_data['categories'].keys()))
        self.groundtruths = self.split_areaRng(GT_data['annotations'])


    def __call__(self, predictions):
        all_preds_dict = self.transform_pred_format(predictions)
        all_preds_dict = self.cutoff_maxDets(all_preds_dict, maxDets=self.maxDets)
        detections = self.split_areaRng(all_preds_dict)

        mAP_info = {}
        for areaLbl in self.areaRngLbl:
            AP_info_per_class = []
            for c in self.classes_index:
                res = self.calculate_AP_single_class(groundtruths=self.groundtruths[areaLbl],
                                                     detections=detections[areaLbl],
                                                     class_id=c)
                AP_info_per_class.append(res)
            mAP_info[areaLbl] = self.calculate_mAP(AP_info_per_class)
        
        eval_text = '\n'
        for areaLbl in self.areaRngLbl:
            if areaLbl == 'all':
                eval_text += self.summarize(mAP_info[areaLbl]['mAP05095'], iouThr=None, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_info[areaLbl]['mAP05'], iouThr=0.5, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
                eval_text += self.summarize(mAP_info[areaLbl]['mAP075'], iouThr=0.75, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
            else:
                eval_text += self.summarize(mAP_info[areaLbl]['mAP05'], iouThr=0.5, areaLbl=areaLbl, maxDets=self.maxDets)
                eval_text += '\n'
        return mAP_info, eval_text


    def summarize(self, mAP, iouThr=None, areaLbl='all', maxDets=100):
        iStr = '\t - {:<16} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision'
        typeStr = '(AP)'
        iouStr = '{:0.2f}:{:0.2f}'.format(self.iouThrs[0], self.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)
        return (iStr.format(titleStr, typeStr, iouStr, areaLbl, maxDets, mAP))


    def calculate_mAP(self, AP_info_per_class):
        AP05_per_class = {}
        num_positives_per_class, num_TP_per_class, num_FP_per_class = {}, {}, {}
        mAP05, mAP075, mAP05095 = 0, 0, 0
        valid_num_classes = 1e-10

        for res in AP_info_per_class:
            if res['total_positives'] > 0:
                valid_num_classes += 1 
                AP05_per_class[res['class']] = res['AP05']
                num_positives_per_class[res['class']] = res['total_positives']
                num_TP_per_class[res['class']] = res['total_TP']
                num_FP_per_class[res['class']] = res['total_FP']
                mAP05 += res['AP05']
                mAP075 += res['AP075']
                mAP05095 += res['AP05095']
        mAP05 /= valid_num_classes
        mAP075 /= valid_num_classes
        mAP05095 /= valid_num_classes

        res = {
            'AP05_PER_CLASS': AP05_per_class,
            'num_positives_per_class': num_positives_per_class,
            'num_TP_per_class': num_TP_per_class,
            'num_FP_per_class': num_FP_per_class,
            'mAP05': mAP05,
            'mAP075': mAP075,
            'mAP05095': mAP05095,
        }
        return res


    def calculate_AP_single_class(self, groundtruths, detections, class_id):
        GTs_per_class = [g for g in groundtruths if g['class_id'] == class_id]
        preds_per_class = [d for d in detections if d['class_id'] == class_id]
        preds_per_class = sorted(preds_per_class, key=lambda x:x['confidence'], reverse=True)

        num_true = len(GTs_per_class)
        num_positive = len(preds_per_class)
        TP = np.zeros(shape=(len(self.iouThrs), num_positive))
        FP = np.zeros(shape=(len(self.iouThrs), num_positive))

        if num_positive == 0:
            res = {
            'class': class_id,
            'precision05': 0,
            'recall05': 0,
            'AP05': 0,
            'AP075': 0,
            'AP05095': 0,
            'total_positives': num_positive,
            'total_TP': np.sum(TP),
            'total_FP': np.sum(FP)
            }
            return res

        flag_GTs_per_image = Counter(g['image_id'] for g in GTs_per_class)
        for k, v in flag_GTs_per_image.items():
            flag_GTs_per_image[k] = np.zeros(shape=(len(self.iouThrs), v))

        for i in range(len(preds_per_class)):
            pred_in_image = preds_per_class[i]
            GT_in_image = [gt for gt in GTs_per_class if gt['image_id'] == pred_in_image['image_id']]

            iou_max = 0
            for j in range(len(GT_in_image)):
                iou = self.get_IoU(pred_in_image['bbox'], GT_in_image[j]['bbox'])
                if iou > iou_max:
                    iou_max = iou
                    jmax = j
            
            for k in range(len(self.iouThrs)):
                if iou_max >= self.iouThrs[k]:
                    if flag_GTs_per_image[pred_in_image['image_id']][k, jmax] == 0:
                        TP[k, i] = 1
                        flag_GTs_per_image[pred_in_image['image_id']][k, jmax] = 1
                    else:
                        FP[k, i] = 1
                else:
                    FP[k, i] = 1

        acc_FP = np.cumsum(FP, axis=1)
        acc_TP = np.cumsum(TP, axis=1)
        rec = acc_TP / (num_true + 1e-10)
        prec = np.divide(acc_TP, (acc_FP + acc_TP))

        APs = []
        AP05095 = 0
        for idx in range(len(self.iouThrs)):
            ap = self.calculate_average_precision(rec[idx], prec[idx])
            APs.append(ap)
            AP05095 += (self.iouThrs[idx] * ap)
        AP05095 /= sum(self.iouThrs)

        res = {
            'class' : class_id,
            'precision05' : list(prec[0].round(4)),
            'recall05' : list(rec[0].round(4)),
            'AP05' : APs[0],
            'AP075' : APs[5],
            'AP05095' : AP05095,
            'total_positives' : num_positive,
            'total_TP' : np.sum(TP),
            'total_FP' : np.sum(FP)
        }
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

    
    def cutoff_maxDets(self, predictions, maxDets):
        image_ids = set(map(lambda x:x['image_id'], predictions))
        dets_per_image = [[item for item in predictions if item['image_id']==x][:maxDets] for x in image_ids]
        new_predictions = []
        for item in dets_per_image:
            new_predictions.extend(item)
        return new_predictions


    def transform_pred_format(self, predictions):
        new_predictions = []

        for prediction in predictions:
            filename, pred_yolo = prediction
            img_id = self.image_to_info[filename]['image_id']
            img_h = self.image_to_info[filename]['height']
            img_w = self.image_to_info[filename]['width']

            pred_voc = pred_yolo.copy()
            pred_voc[:, 1:5] = box_transform_xcycwh_to_x1y1x2y2(pred_voc[:, 1:5]/self.input_size)
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

    def calculate_average_precision(self, rec, prec):
        mrec = [0] + [e for e in rec] + [1]
        mpre = [0] + [e for e in prec] + [0]

        for i in range(len(mpre)-1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])

        ii = []
        for i in range(len(mrec)-1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i+1)

        ap = 0
        for i in ii:
            ap += np.sum((mrec[i] - mrec[i-1]) * mpre[i])
        return ap