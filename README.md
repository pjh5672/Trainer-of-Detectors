# <div align="center">Object Detector Trainer</div>

## Description

This is repository for source code to train various object detection models. currently, It supports pretrained YOLOv3 weight(yolov3.pt) from [here](https://drive.google.com/drive/folders/15qXxbD7RG19uZBhG3NPWwfqt6OdksAPR?usp=sharing)
**Prototyping now...**

**COCO2017 Average Precision**
| Model | size<sup>(pixels) | mAP<sup>0.5:0.95 | mAP<sup>0.5 |
| :---: | :---: | :---: | :---: | 
| YOLOv3 (paper) | 320 | 28.7 | 51.8 |
| YOLOv3 (paper) | 416 | 31.2 | 55.4 |
| YOLOv3 (paper) | 512 | 32.7 | 57.7 |
| YOLOv3 (paper) | 608 | 33.1 | 58.2 |

- **Installation of third-party package library (CUDA11.3)**
```bash
$ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

## Update

<details>
    <summary><b> Timeline in 2022 </b></summary>

| Date | Content |
|:----:|:-----|
| 08-21 | add:consider class conditional probability & support yolov3.pt weight |
| 08-20 | fix:chnage BCELoss -> BCEWithLogitLoss due to stability in case of AMP computation |
| 08-17 | bug:sanity check for avoiding CUDA runtime error(device-side assert triggered) during training |
| 08-10 | add:visualize functions for PR curve, AP@0.50, num of detection rate(TP, FP, FN) per class |
| 08-09 | fix:mAP calculation optimization x150 speed up and process await delay reduction with DDP training |
| 08-07 | add:learning rate scheduler (160 epochs with starting inital lr:0.001, dividing it by 10 at 30, 60 epochs) |
| 08-04 | fix:code refactoring (visualizer for prediction of letter box) |
| 08-03 | fix:code refactoring (del redundant functions) |
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

## Contact

- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  