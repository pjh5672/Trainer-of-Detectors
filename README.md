# <div align="center">Object Detector Trainer</div>

## <div align="center">Description</div>

This is repository for source code to train various object detection models.  currently, It supports YOLOv3 model training with Darknet53(You can download darknet53 weight from [here](https://drive.google.com/file/d/1VuV0llnEPUGiPq-cKhfSQlKQ7Zg6jlp9/view?usp=sharing))
**Prototyping now...**

**COCO2017 Average Precision at IoU=0.5 (AP50)**

| Model | size<sup>(pixels) | mAP<sup>0.5:0.95 | mAP<sup>0.5 |
| :---: | :---: | :---: | :---: | 
| YOLOv3 (paper) | 320 | 28.7 | 51.8 |
| YOLOv3 (paper) | 416 | 31.2 | 55.4 |
| YOLOv3 (paper) | 512 | 32.7 | 57.7 |
| YOLOv3 (paper) | 608 | 33.1 | 58.2 |


## <div align="center">Updates</div>

#### <details><summary><b> Timeline in 2022 </b></summary>  

| Date | Content |
|:----:|:-----|
| 08-02 | add:code integration of YOLOv3 trainer supporting Linux(Multi-GPUs) & Windows(Single-GPU) |
| 08-01 | add:torch DistributedDataParallel(DDP) model train function on multi-GPUs |
| 07-30 | fix:loss function, mAP calculate error debug when validation mode |
| 07-28 | add:mAP evaluation function, mAP logging, basic augmentation implementation |
| 07-12 | add:COCO evaluation API test env initial build |
| 07-11 | add:Best Possible Recalls(BPR) implementation |
| 07-07 | fix:valid loss function, valid loss for running with no object |
| 07-05 | fix:yolov3 loss function |
| 07-04 | First commit |

</details>

## <div align="center">Contact</div>

- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  