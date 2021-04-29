import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(args, is_train=True, visualize=False):
    if is_train:
        tfms = [
            alb.Transpose(p=args["transpose"]),
            alb.HorizontalFlip(p=args["hflip"]),
            alb.VerticalFlip(p=args["vflip"]),
            alb.RandomBrightness(**args["brightness"]),
            alb.RandomContrast(**args["contrast"]),
            alb.CenterCrop(
                args["image_size"], args["image_size"], p=args["center_crop"]
            ),
            alb.Resize(args["image_size"], args["image_size"]),
        ]
    else:
        tfms = [
            alb.CenterCrop(
                args["image_size"], args["image_size"], p=args["center_crop"]
            )
        ]
        tfms = [alb.Resize(args["image_size"], args["image_size"])]

    if not visualize:
        tfms += [
            alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
            ToTensorV2(),
        ]

    return alb.Compose(tfms)
