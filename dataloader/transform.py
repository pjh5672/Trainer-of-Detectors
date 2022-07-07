import albumentations as album
from albumentations.pytorch import ToTensorV2


def build_transformer(image_size=(448, 448)):
    transformers = {}
    # transformers['train'] = A.Compose([
    #     A.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=(127,127,127), p=0.5),
    #     A.RandomSizedBBoxSafeCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
    #     A.HorizontalFlip(p=0.5),
    #     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
    #     A.RandomBrightnessContrast(p=0.3),
    #     A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    #     ToTensorV2()],
    #     bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']),
    # )

    transformers['train'] = album.Compose([
        album.HorizontalFlip(p=0.5),
        album.Resize(height=image_size[0], width=image_size[1]),
        album.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2()],
        bbox_params=album.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    
    transformers['val'] = album.Compose([
            album.Resize(height=image_size[0], width=image_size[1]),
            album.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
            ToTensorV2()],
        bbox_params=album.BboxParams(format='yolo', label_fields=['class_ids']),
    )
    return transformers