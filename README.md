# <div align="center">Object Detector Trainer</div>

## Description

This is repository for source code to train various object detection models. currently, It supports pretrained YOLOv3 weight(yolov3.pt) from [here](https://drive.google.com/drive/folders/15qXxbD7RG19uZBhG3NPWwfqt6OdksAPR?usp=sharing)

 - **COCO2017 Average Precision** (sanity check of training code - target model: YOLOv3)

| Model | size<sup>(pixels) | mAP<sup>0.5:0.95 | mAP<sup>0.5 |
| :---: | :---: | :---: | :---: | 
| YOLOv3(paper) | 320 x 320 | 28.7 | 51.8 |
| YOLOv3(paper) | 416 x 416 | 31.2 | 55.4 |
| YOLOv3(paper) | 512 x 512 | 32.7 | 57.7 |



### Usage

#### 1. Data Config

 - You can copy `*.yaml.example` to `*.yaml` and use it as a training argument.
 - **`*.yaml` Arguments**
    - **Path** : path to the directory containing the dataset
    - **TRAIN** : path where training images are stored
    - **VAL** : path where the image for verification is stored
    - **mAP_FILE** : path of verification data file to be loaded for mAP metric calculation (automatically created when verification data is first loaded)
    - **NAMES** : list of category names the model will learn from

#### 2. Train Config

 - You can copy `*.yaml.example` to `*.yaml` and use it as a training argument.
 - **`*.yaml` Arguments**
    - **RESUME_PATH** : checkpoint path to be loaded when continuing training on a model that has stopped training (ckechpoint consists of model_state_dict, optimizer_state_dict, epoch)
    - **PRETRAINED_PATH** : path of pre-trained weights file (only model_state_dict is wrapped)
    - **NUM_EPOCHS** : number of epochs to train the model
    - **INPUT_SIZE** : size of input image to be used for model calculation
    - **BATCH_SIZE** : size of the mini-batch to be calculated during one iteration of training
    (...skipped)


## Usage

#### 1. Train Detector

 - **Train Arguments**
    - **data_path** : path to data.yml file
    - **config_path** : path to config.yml file
    - **exp_name** : name to log training
    - **gpu_ids** : list of GPU IDs
    - **img_interval** : image logging interval
    - **start_save** : starting model saving epoch
    - **init_score** : initial mAP score for update best model    

```python
# simple example on parallel training on 2 GPUs
train.py --data_path data/coco128.yaml --config config/yolov3.yaml --exp_name train --gpu_ids 0 1
```


#### 2. Result Analysis

 - **Log file**
```log
2022-08-23 13:41:49 | Rank 0 | [TRAIN] hash: 19314466396 version: 2022-08-04_18-17 
2022-08-23 13:41:49 | Rank 0 | [VAL] hash: 814705164 version: 2022-08-04_18-17 
2022-08-23 13:41:53 | Rank 0 | Path to pretrained model: ./weights/yolov3.pt

2022-08-23 14:05:38 | Rank 0 | [Epoch:001/300] Train Loss: 173.23, Val Loss: 179.45
2022-08-23 14:06:03 | Rank 0 | [Epoch:001/300] mAP Computation Time(sec): 25.7212
2022-08-23 14:06:03 | Rank 0 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.070
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.160
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.051
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.024
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.089
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.228

                                        ...

2022-08-24 17:03:50 | Rank 0 | [Epoch:053/300] Train Loss: 125.79, Val Loss: 145.82
2022-08-24 17:04:19 | Rank 0 | [Epoch:053/300] mAP Computation Time(sec): 29.5476
2022-08-24 17:04:19 | Rank 0 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.144
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.280
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.131
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.044
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.199
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.395
```


 - **PR curve per category**
   <div align="center">
   <a href=""><img src=./asset/PR_curve/airplane.png width="19%" /></a>
   <a href=""><img src=./asset/PR_curve/boat.png width="19%" /></a>
   <a href=""><img src=./asset/PR_curve/car.png width="19%" /></a>
   <a href=""><img src=./asset/PR_curve/dog.png width="19%" /></a>
   <a href=""><img src=./asset/PR_curve/person.png width="19%" /></a>
   </div>


 - **AP score per category & Detection rate**
   <div align="center">
   <a href=""><img src=./asset/figure-AP.png width="49%" /></a>
   <a href=""><img src=./asset/figure-dets.png width="49%" /></a>
   </div>
 

 - **Image file**
    - train image(top row) and validation result(bottom row) at 10,20,30,40,50 epoch
   <div align="center">
   <a href=""><img src=./asset/images/train/EP010.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/train/EP020.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/train/EP030.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/train/EP040.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/train/EP050.jpg width="15%" /></a>
   </div>
   <div align="center">
   <a href=""><img src=./asset/images/val/EP010.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/val/EP020.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/val/EP030.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/val/EP040.jpg width="15%" /></a>
   <a href=""><img src=./asset/images/val/EP050.jpg width="15%" /></a>
   </div>


---

- **Third-party package library Installation (with CUDA11.3)**
```bash
$ pip install -r requirements.txt
```

## Update

<details>
    <summary><b> Timeline in 2022 </b></summary>

| Date | Content |
|:----:|:-----|
| 08-24 | add:update README.md file |
| 08-22 | add:train with resume mode in case of previous models |
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