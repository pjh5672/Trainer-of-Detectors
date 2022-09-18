# <div align="center">Object Detector Trainer</div>

---

## [Contents]
1. [Description](#description)  
  1-1. [Data Configuraion](#data-configuraion)  
  1-2. [Train Configuraion](#train-configuraion)  
2. [Usage](#usage)  
  2-1. [Train Detector](#train-detector)  
  2-2. [Validate mAP metric](#validate-map-metric)  
  2-3. [Analyse Result](#analyse-result)  
3. [Update](#update)   
4. [Contact](#contact)

---

## [Description]

This is repository for source code to train various object detection models. currently, You can download related weight files from [here](https://drive.google.com/drive/folders/15qXxbD7RG19uZBhG3NPWwfqt6OdksAPR?usp=sharing)


 - **Performance Benchmark**

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | mAP<br><sup>(@0.5:0.95) | mAP<br><sup>(@0.5) | Params<br><sup>(M) | FLOPS<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| YOLOv3<br><sup>(<u>Paper:page_with_curl:</u>)</br> | MS-COCO | train2017 | val2017 | 416 | 31.2 | 55.4 | 61.95 | 65.86 |
| YOLOv3<br><sup>(<u>Our:star:</u>)</br> | MS-COCO | train2017 | val2017 | 416 | 26.0 | 44.3 | 61.95 | 66.17 |
| YOLOv3<br><sup>(<u>Paper:page_with_curl:</u>)</br> | Pascal VOC | trainval2007+2012| test2007 | 416 | - | 76.5 | 61.63 | 65.86 |
| YOLOv3<br><sup>(<u>Our:star:</u>)</br> | Pascal VOC | trainval2007+2012 | test2007 | 416 | 51.6 | 77.5 | 61.63 | 65.74 |


### Data Configuraion

<details>
<summary> Data.yaml Argument </summary>

  - You can copy *.yaml.example to *.yaml and use it as a training argument
  - **`PATH`** : path to the directory containing the dataset
  - **`TRAIN`** : path where training images are stored
  - **`VAL`** : path where the image for verification is stored
  - **`mAP_FILE`** : path of verification data file to be loaded for mAP metric calculation (automatically created when verification data is first loaded)
  - **`NAMES`** : list of category names the model will learn from

</details>

### Train Configuraion

<details>
<summary> Config.yaml Argument </summary>

  - You can copy `*.yaml.example` to `*.yaml` and use it as a training argument  
  - **Weight Parameter**
    - **`RESUME_PATH`** : checkpoint path to be loaded when continuing training on a model that has stopped training (ckechpoint consists of model_state_dict, optimizer_state_dict, epoch)
    - **`PRETRAINED_PATH`** : path of pre-trained weights file (only model_state_dict is wrapped)

  - **Train Parameter**
    - **`NUM_EPOCHS`** : number of epochs to train the model
    - **`INPUT_SIZE`** : size of input image (H,W) to be used for model calculation
    - **`INPUT_CHANNEL`** : size of input channel to be used for model calculation
    - **`BATCH_SIZE`** : size of the mini-batch to be calculated during one iteration of training (recommend setting 128 batch-size or less)
    - **`INIT_LEARNING_RATE`** : initial learning rate (recommend setting for SGD: 0.01, Adam: 0.001)
    - **`FINAL_LEARNING_RATE`** : final learning rate
    - **`WEIGHT_DECAY`** : optimizer weight decay
    - **`MOMENTUM`** : momentum in SGD/beta1 in Adam optimizer
    - **`WARMUP_EPOCH`** : warmup epochs for stable initial training
    - **`WARMUP_MOMENTUM`** : warmup initial momentum
    - **`WARMUP_BIAS_LR`** : warmup initial bias lr
    - **`GET_PBR`** : mode on/off for calculate possible best recalls
    - **`ANCHOR_IOU_THRESHOLD`** : minimum threshold of overlap size with the predefined anchors to transform into learnable targets
    - **`ANCHORS`** : the width and height of the predefined anchor boxes for small/medium/large scale

  - **Augment Parameter**
    - **`FLIP_UD`** : Probability for flipping up/down of data
    - **`FLIP_LR`** : Probability for flipping left/right of data
    - **`HSV_H`** : Random distribution range to change the color of hue in hsv
    - **`HSV_S`** : Shift limit for change of hue in HSV color space 
    - **`HSV_V`** : Shift limit for change of value in HSV color space 
    - **`ROTATE`** : Shift limit for change of saturation in HSV color space 
    - **`SHIFT`** : Limit the range of moving up, down, left, and right
    - **`SCALE`** : Limit the range to zoom in/out of data
    - **`PERSPECTIVE`** : Degree Limit to which perspective transformation is applied
    - **`MIXUP`** : Probability for applying mixup augmentation

  - **AP Metric Parameter**
    - **`MAX_DETS`** : maximum number of predictions per a frame
    - **`MIN_SCORE_THRESH`** : minimum threshold to filter out predictions by confidence score
    - **`MIN_IOU_THRESH`** : minimum threshold of overlap size to merge out predictions by Non-Maximum Suppression

  - **Loss Parameter**
    - **`IGNORE_THRESH`** : minimum threshold whether to include learning for no-object 
    - **`COEFFICIENT_COORD`** : gain of boxes regression loss to be included in learning loss
    - **`COEFFICIENT_NOOBJ`** : gain of no-object entropy loss to be included in learning loss

</details>


## [Usage]

### Train Detector

<details>
<summary> Training Argument </summary>

  - **`data`** : path to data.yaml file
  - **`config`** : path to config.yaml file
  - **`exp_name`** : name to log training
  - **`world_size`** : number of available GPU devices
  - **`img_interval`** : image logging interval
  - **`start_eval`** : starting epoch for mAP evaluation
  - **`linear_lr`** : use of linear LR scheduler (default: one cyclic scheduler)
  - **`no_amp`** : use of FP32 training without AMP (default: AMP training)
  - **`freeze_backbone`** : freeze backbone layers (default: False)
  - **`adam`** : use of Adam optimizer (default: SGD optimizer)

</details>

```python
# simple example on parallel training on 2 GPUs
python train.py --data data/coco128.yaml --config config/yolov3_coco.yaml --exp_name train --world_size 2
```

### Validate mAP metric

<details>
<summary> Validating Argument </summary>

  - **`data`** : path to data.yaml file
  - **`config`** : path to config.yaml file
  - **`model`** : path to trained model weight
  - **`rank`** : GPU device index for running

</details>

```python
# simple example on parallel training on 2 GPUs
python val.py --data data/voc.yaml --config config/yolov3_voc.yaml --model weight/voc_best.pt
```

 - Terminal Output
    ```log
      - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
      - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.720
      - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.535
      - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.184
      - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.508
      - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.766
    ```

### Analyse Result

<details>
<summary> Log file </summary>

```log
2022-09-09 09:53:18 | Rank 0 | [TRAIN] hash: 1780273991 version: 2022-09-09_08-14 
2022-09-09 09:53:18 | Rank 0 | [VAL] hash: 434734692 version: 2022-09-09_08-14 
2022-09-09 09:53:19 | Rank 0 | Params(M): 61.63, FLOPS(B): 65.74
2022-09-09 09:53:22 | Rank 0 | Path to pretrained model: ./weights/yolov3.pt

2022-09-09 09:57:28 | Rank 0 | [Epoch:001/1000] Train Loss: 173.23, Val Loss: 179.45
2022-09-09 10:01:23 | Rank 0 | [Epoch:001/1000] mAP Computation Time(sec): 25.7212
2022-09-09 10:05:18 | Rank 0 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.070
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.160
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.051
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.024
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.089
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.228

                                        ...

2022-09-12 09:32:56 | Rank 0 |  Best mAP@0.5: 0.720 at [Epoch:929/1000]
2022-09-12 09:32:56 | Rank 0 | 
	 - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
	 - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.720
	 - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.535
	 - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = 0.184
	 - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = 0.508
	 - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = 0.766
```

</details>


<details>
<summary> PR curve per category </summary>

<div align="center">
   <a href=""><img src=./asset/PR_curve/aeroplane.png width="16%" /></a>
   <a href=""><img src=./asset/PR_curve/bicycle.png width="16%" /></a>
   <a href=""><img src=./asset/PR_curve/boat.png width="16%" /></a>
   <a href=""><img src=./asset/PR_curve/bus.png width="16%" /></a>
   <a href=""><img src=./asset/PR_curve/car.png width="16%" /></a>
   <a href=""><img src=./asset/PR_curve/cat.png width="16%" /></a>
</div>

</details>


<details>
<summary> AP score per category </summary>

<div align="center">
  <a href=""><img src=./asset/figure-AP_EP929.png width="60%" /></a>
</div>

</details>


<details>
<summary> Detection rate </summary>

<div align="center">
  <a href=""><img src=./asset/figure-dets_EP929.png width="60%" /></a>
</div>

</details>


<details>
<summary> Image file </summary>

<div align="center">
  <a href=""><img src=./asset/images/train/EP100.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/train/EP200.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/train/EP300.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/train/EP400.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/train/EP500.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/train/EP600.jpg width="16%" /></a>
</div>

<div align="center">
  <a href=""><img src=./asset/images/val/EP100.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/val/EP200.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/val/EP300.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/val/EP400.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/val/EP500.jpg width="16%" /></a>
  <a href=""><img src=./asset/images/val/EP600.jpg width="16%" /></a>
</div>

<div align="center">
  - Train image(top row) and validation result(bottom row) at 100, 200, 300, 400, 500, 600 epoch
</div>

</details>


---

- **Third-party package library Installation (with CUDA11.3)**
```bash
$ pip install -r requirements.txt
```

## [Update]

<details>
    <summary> Timeline in 2022 </summary>

| Date | Content |
|:----:|:-----|
| 09-18 | fix:exceed baseline performance(mAP) on VOC dataset |
| 09-17 | add:upload validation files for mAP calculation on voc, coco2017 dataset |
| 09-16 | add:non-maximum suppression with multi-class & class-agnostic |
| 09-15 | add:val.py for reproducing mAP with trained model |
| 09-14 | add:data augmentation with perspective transformation, random crop, mixup |
| 09-09 | fix:VOC dataset  change for paper performance reproducing |
| 09-07 | fix:resume mode in DDP |
| 09-06 | fix:make model training stable with adjust lr in early training,loss accumulate mode |
| 09-05 | add:data augmentation(albumentation, fliplr, random perspective transform) |
| 09-04 | add:pretrained yolov3 weights excluding head update, fix:mae & bce loss nan due to large batch size |
| 09-03 | add:PASCAL-VOC2012 data update, More than 20 figures memory comsumption warning |
| 08-27 | add:exception visualize condition in case of detection nothing |
| 08-26 | add:logging function for model parameters & FLOPS |
| 08-25 | add:automatic mixed precision applied & log argument command function |
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


## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  