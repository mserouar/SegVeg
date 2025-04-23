![concatenated_output](https://github.com/user-attachments/assets/6e38bbcb-0868-495e-8099-7b21c465c229)

# Vegetation Segmentation with VegAnn

This notebook shows how to apply semantic segmentation with the Unet model tuned on the VegAnn dataset for RGB image segmentation into Vegetation/Background pixel classification.

Madec, S., Irfan, K., Velumani, K. et al. VegAnn, Vegetation Annotation of multi-crop RGB images acquired under diverse conditions for segmentation. Sci Data 10, 302 (2023). https://doi.org/10.1038/s41597-023-02098-y

Author : Simon Madec + Mario Serouart (paper | notebook)

# Senescent Segmentation with SegVeg

This section shows how to apply semantic segmentation with the SVM | XGBoost model tuned on the SegVeg dataset for RGB image segmentation into Green Vegetation/Senescent Vegetation pixel classification.

Serouart Mario Madec Simon David Etienne Velumani Kaaviya Lopez Lozano Raul Weiss Marie Baret Frédéric. SegVeg: Segmenting RGB Images into Green and Senescent Vegetation by Combining Deep and Shallow Methods. Plant Phenomics (2022). https://doi.org/10.34133/2022/9803570.

Authors : Mario Serouart | Simon Madec (paper)


# Senescent + Necrosis Segmentation with improved SegVeg by ETHZ

This section shows how to apply semantic segmentation with the improved model tuned on the SegVeg dataset for RGB image segmentation into Green Vegetation/Senescent and Necrotic Vegetation pixel classification. It follows the same methodological principles, such as the use of feature extraction and color space transformations.

Jonas Anderegg, Radek Zenkl, Achim Walter, Andreas Hund, Bruce A. McDonald. Combining High-Resolution Imaging, Deep Learning, and Dynamic Modeling to Separate Disease and Senescence in Wheat Canopies. Plant Phenomics (2023). https://doi.org/10.34133/plantphenomics.0053.

And: https://github.com/and-jonas/wheat-segmentation-models?tab=readme-ov-file

Authors : Jonas Anderegg  (paper | notebook) + Mario Serouart (notebook)


# Information

⚠️ If you find this work useful in your research (Python module, models or Dataset), please cite **ALL** mentioned papers above.

📚 All data needed to run this notebook/markdown are available in **Realeases** SegVeg Data (v1.1.0)


# Install libraries


```python
!pip install torch
!pip install git+https://github.com/PyTorchLightning/pytorch-lightning
!pip install segmentation_models_pytorch
!pip install xgboost==1.5.2
```

# Import librairies and VegAnn Classes/functions :


```python
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
from typing import Dict, List

# import logging
# import pathlib
# import pandas as pd
# import os
# import glob
# import numpy as np
# import PIL
# from PIL import Image, ImageCms
# import pickle
# from tqdm import tqdm
# import cv2
# from skimage import color, io
# import time
# import json
# from pathlib import Path

import ast
import glob
import logging
import os
import pickle
import warnings
from pathlib import Path


class VegAnnModel(pl.LightningModule):
    def __init__(self, arch: str, encoder_name: str, in_channels: int, out_classes: int, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,)

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.train_outputs, self.val_outputs, self.test_outputs = [], [], []

    def forward(self, image: torch.Tensor):
        # normalize image here #todo
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch: Dict, stage: str):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs: List[Dict], stage: str):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        dataset_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_f1": per_image_f1,
            f"{stage}_dataset_f1": dataset_f1,
            f"{stage}_per_image_acc": per_image_acc,
            f"{stage}_dataset_acc": dataset_acc,
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)

    def training_step(self, batch: Dict, batch_idx: int):
        step_outputs = self.shared_step(batch, "train")
        self.train_outputs.append(step_outputs)
        return step_outputs

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.train_outputs, "train")
        self.train_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int):
        step_outputs = self.shared_step(batch, "valid")
        self.val_outputs.append(step_outputs)
        return step_outputs

    def on_validation_epoch_end(self, *args, **kwargs):
        self.shared_epoch_end(self.val_outputs, "valid")
        self.val_outputs = []

    def test_step(self, batch: Dict, batch_idx: int):
        step_outputs = self.shared_step(batch, "test")
        self.test_outputs.append(step_outputs)
        return step_outputs

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_outputs, "test")
        self.test_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def colorTransform_VegGround(im,X_true,alpha_vert,alpha_g):
    alpha = alpha_vert
    color = [97,65,38]
    # color = [x / 255 for x in color]
    image=np.copy(im)
    for c in range(3):
        image[:, :, c] =np.where(X_true == 0,image[:, :, c] *(1 - alpha) + alpha * color[c] ,image[:, :, c])
    alpha = alpha_g
    color = [34,139,34]
    # color = [x / 255 for x in color]
    for c in range(3):
        image[:, :, c] =np.where(X_true == 1,image[:, :, c] *(1 - alpha) + alpha * color[c] ,image[:, :, c])
    return image

print("VegAnn Class built ✅")
```

    VegAnn Class built ✅


# Load weights of VegAnn model


```python
ckt_path = "/content/VegAnn.ckpt"

checkpoint = torch.load(ckt_path, map_location=torch.device('cpu'))
model = VegAnnModel("Unet","resnet34",in_channels = 3, out_classes=1 )
model.load_state_dict(checkpoint["state_dict"])
preprocess_fn = smp.encoders.get_preprocessing_fn("resnet34", pretrained= "imagenet")
model.eval(); # Remove ';' if raised issue, it just avoids printing model summary

print("VegAnn model loaded ✅")
```

    VegAnn model loaded ✅


# Predict and Visualize


```python
# Function that add mirrored border
# Adding Context/Features helps to better segmentate borders and avoiding artefacts due to patchification
# The mirrored image is not plotted
def add_miror_offset(image: np.ndarray, offset: int):
    """
    create a border around the image
    The border will be mirror reflection of the border elements.
    """
    img = cv2.copyMakeBorder(
        image,
        top=offset,
        bottom=offset,
        left=offset,
        right=offset,
        borderType=cv2.BORDER_REFLECT,)
    return img

# Obviously, you have to remove this artificially created border
def remove_offset(img: np.ndarray, offset: int) -> np.ndarray:
    """
    remove offset from top, both, left, right
    """
    if len(img.shape) == 2:
        height, width = img.shape
        processed_image = img[offset : height - offset, offset : width - offset]
    elif len(img.shape) == 3:
        height, width, _ = img.shape
        processed_image = img[offset : height - offset, offset : width - offset, :]
    else:
        raise ValueError("img must be a 2d or a 3d array")
    return processed_image


def rgb_replacement(
    image: np.ndarray,
    index_mask: np.ndarray,
    mask_value: int,
    channels_values: tuple[int]):
    """
    allow channel attribution for 3 channels masks
    """
    c1, c2, c3 = cv2.split(image)
    v1, v2, v3 = channels_values
    c1[index_mask == mask_value] = v1
    c2[index_mask == mask_value] = v2
    c3[index_mask == mask_value] = v3
    img = cv2.merge((c1, c2, c3))

    return img


def visualization(
    img: np.ndarray, mask: np.ndarray, ignore_classes: list[int] = [0]) -> tuple[np.ndarray, list, list]:
    """
    take an image and corresponding mask and output 3 channel
    rgb visualisation with class color & transparency
    """
    color_vizu = []
    index_class = []
    mask3c = img.copy()
    classes = list(np.unique(mask))
    for c in classes:
        if c not in ignore_classes:
            mask3c = rgb_replacement(mask3c, mask, c, __COLORS[c])
            color_vizu.append(__COLORS[c])
            index_class.append(c)

    mask3c, img = mask3c.astype("uint8"), img.astype("uint8")
    visualization_image = cv2.addWeighted(img, 0.5, mask3c, 0.5, 0)

    return visualization_image, index_class, color_vizu


def Text_Image(
    rgb_image: np.ndarray, text: str, Stg: str, index_class: int, color_vizu: np.ndarray
    ) -> np.ndarray:
    """
    Add some text and colors to a Visualisation image
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    Pos_txt = (15, 40)
    fontScale = 0.75
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 0

    cv2.putText(
        rgb_image, text, Pos_txt, font, fontScale, fontColor, thickness, lineType
    )

    Pos_rect = (15, 50)
    if Stg == "VegAnn":
      for c in color_vizu:
        cv2.rectangle(
            rgb_image, Pos_rect, (Pos_rect[0] + 20, Pos_rect[1] + 20), c, -1
        )
        cv2.putText(
            rgb_image,
            "Whole Vegetation",
            (Pos_rect[0] + 35, Pos_rect[1] + 18),
            font,
            fontScale,
            c,
            thickness,
            lineType,
        )

    if Stg == "SegVeg_XG_SVM":
      labels = ["Sol", "Healthy", "Chlorosis (Senescent)"]
      id2label = {idx: c for idx, c in enumerate(labels)}
      idx = 0
      for c in index_class:
          cv2.rectangle(
              rgb_image,
              Pos_rect,
              (Pos_rect[0] + 20, Pos_rect[1] + 20),
              color_vizu[idx],
              -1,
          )
          cv2.putText(
              rgb_image,
              str(id2label[int(c)]),
              (Pos_rect[0] + 35, Pos_rect[1] + 18),
              font,
              fontScale,
              color_vizu[idx],
              thickness,
              lineType,
          )

          idx = idx + 1
          Pos_rect = (Pos_rect[0], Pos_rect[1] + 25)

    if Stg == "SegVeg_Necrosis":
      labels = ["Sol", "Healthy", "Necrosis (Dead)", "Chlorosis (Senescent)"]
      id2label = {idx: c for idx, c in enumerate(labels)}
      idx = 0
      for c in index_class:
          cv2.rectangle(
              rgb_image,
              Pos_rect,
              (Pos_rect[0] + 20, Pos_rect[1] + 20),
              color_vizu[idx],
              -1,
          )
          cv2.putText(
              rgb_image,
              str(id2label[int(c)]),
              (Pos_rect[0] + 35, Pos_rect[1] + 18),
              font,
              fontScale,
              color_vizu[idx],
              thickness,
              lineType,
          )

          idx = idx + 1
          Pos_rect = (Pos_rect[0], Pos_rect[1] + 25)

    return rgb_image


# Custom cmap
__COLORS = [(245, 0, 177),
    (0, 255, 244),
    (245, 232, 0),
    (136, 0, 209)]

image = cv2.cvtColor(cv2.imread("/content/test2.png"), cv2.COLOR_BGR2RGB)
image_mirorred = add_miror_offset(image, 50)

preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
image_prcd = preprocess_input(image_mirorred)
image_prcd = image_prcd.astype('float32')

inputs = torch.tensor(image_prcd)
inputs = inputs.permute(2, 0, 1)
inputs = inputs[None,:,:,:]
logits = model(inputs)
pr_mask = logits.sigmoid()

pred = (pr_mask > 0.5).numpy().astype(np.uint8)
pred = np.squeeze(pred)
pred = remove_offset(pred, 50)

visualisation_image_Stg1, index_class, color_vizu = visualization(
    image, pred
)

visualisation_image_Stg1 = Text_Image(
    visualisation_image_Stg1,
    "VegAnn Whole Vegetation Segmentation",
    "VegAnn",
    index_class,
    color_vizu,
)

concatenated_image = np.concatenate((image, visualisation_image_Stg1), axis=1)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.imshow(concatenated_image)
plt.axis('off')
plt.show()

print("VegAnn model succeeded ✅")
```


![concatenated_output_vg](https://github.com/user-attachments/assets/61ce2c4d-9fe4-43e2-afb7-e534acff710b)


    VegAnn model succeeded ✅


# SegVeg

# Loading trained models


```python
# load the Original model (from published SegVeg paper)
model_path = "/content/model_scikit"
model_SVM = pickle.load(Path(model_path).open("rb"))

# load the Original model (from published SegVeg paper) BUT optimized through XGBoost (same data, but reduced computational time)
model_path = "/content/XGBoost"
model_XG = pickle.load(Path(model_path).open("rb"))
new_attrs = ['grow_policy', 'max_bin', 'eval_metric', 'callbacks', 'early_stopping_rounds', 'max_cat_to_onehot', 'max_leaves', 'sampling_method', 'enable_categorical', 'feature_types', 'max_cat_threshold', 'predictor']
for attr in new_attrs:
    setattr(model_XG, attr, None)

# load the Original model (from published SegVeg paper) BUT improved thanks to Jonas Anderegg et al. work.
# No more confusion between Chlorosis-Yellow and Necrosis-Brown
model_path = "/content/Necrosis.pkl"
model_Necrosis = pickle.load(Path(model_path).open("rb"))
```

# Import utils functions


```python
#####################################################################
############ Related to model_SVM and model_XG inference ############
#####################################################################
import xgboost as xgb

def get_features(image) :

    pil_image = Image.fromarray(image)

    hsv = np.array(pil_image.convert(mode='HSV'))
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = np.array(ImageCms.applyTransform(pil_image, rgb2lab))
    ycbcr = np.array(pil_image.convert(mode='YCbCr'))
    Labb = color.rgb2lab(image)
    r = image[:, :,0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    h = (hsv[:, :, 0].astype(np.float32) * 360) / 255
    s = (hsv[:, :, 1]) / 2.55
    a = Labb[:, :, 1]
    bb = Lab[:, :, 2]
    ge =  np.mean([r,g,b], axis = 0)

    CMYlist = [1 - r / 255, 1 - g / 255, 1 - b / 255]
    CMYlist = np.array([np.min(idx) for idx in zip(*CMYlist)])
    m = ((1 - g / 255 - CMYlist ) / (1 - CMYlist )) * 100
    ye = ((1 - b / 255 - CMYlist ) / (1 - CMYlist )) * 100

    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    i = (0.596 * r - 0.275 * g - 0.321 * b)
    q = (0.212 * r - 0.523 * g + 0.311 * b)

    model_input = np.stack((r,g,b,h,s,a,bb,ge,m,ye,cb,cr,i,q), axis=2).squeeze(0)
    model_input = np.nan_to_num(model_input) # Handle black pixels (avoiding Input contains NaN, infinity or a value too large for dtype('float64') error)

    return model_input

    ### Old Parallelise .predict() method ###
    """
    n_cores = config['model_parameters']['n_cores']
    n_samples = X.shape[0]
    slices = [(int(n_samples*i/n_cores), int(n_samples*(i+1)/n_cores)) for i in range(n_cores)]
    data_chunks = [X[i[0]:i[1]] for i in slices]

    if n_cores > 1:
        jobs = (delayed(loaded_model.predict)(array) for array in data_chunks)
        parallel = Parallel(n_jobs=n_cores)

        y_pred  = parallel(jobs)
    else:
        y_pred = [loaded_model.predict(array) for array in data_chunks]

    y_pred  = np.concatenate(y_pred) # Instead of vstack (avoiding all the input array dimensions for the concatenation axis must match exactly error)
    """

def prediction_XG_SVM(image, model, threshold, contrasted, mask) :

    # Revert channels to have RGB /!\ IMPORTANT
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # If Mask doesn't exist create one full of 1's
    if mask is None:
        mask = np.ones((height, width))

    # Flatten arrays
    image = image.reshape((width*height), 1, 3)
    mask = mask.reshape((width*height), 1)

    # Select indices of vegetation pixels
    yellow_green_mask = np.zeros(image.shape[:-1]) # Null array to set bckg pixels to 0
    vegetation_pixels = mask > 0 # Get vegetation pixels indices
    image = image[None, vegetation_pixels]

    # Apply preprocessing to add features to each pixels
    featured_image = get_features(image)

    # Predictions on vegetation pixels
    y_pred = (model.predict_proba(featured_image)[:,1] >= threshold).astype(int)

    # Replace pixels of null array at vegetation indices with output of classification
    yellow_green_mask[vegetation_pixels] = y_pred
    # Reshape
    mask[(mask == 1) & (yellow_green_mask != 1)] = 2
    mask = mask.reshape((height, width))
    # Rotate image

    return mask

#####################################################################
################ Related to model_Necrosis inference ################
#####################################################################

# Function from https://github.com/and-jonas/wheat-segmentation-models
def get_features_Necrosis(image: np.ndarray) -> np.ndarray:

    img_RGB = np.array(image / 255, dtype=np.float32)
    img_RGB = img_RGB[:, :, :3]

    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
    img_YCbCr = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    R, G, B = cv2.split(img_RGB)
    normalizer = np.array(R + G + B, dtype=np.float32)
    normalizer[normalizer == 0] = 10
    r, g, b = (R, G, B) / normalizer

    lambda_r, lambda_g, lambda_b = 670, 550, 480

    TGI = -0.5 * ((lambda_r - lambda_b) * (r - g) - (lambda_r - lambda_g) * (r - b))
    ExR = np.array(1.4 * r - b, dtype=np.float32)
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    descriptors = np.concatenate(
        [
            img_RGB,
            img_HSV,
            img_Lab,
            img_Luv,
            img_YUV,
            img_YCbCr,
            np.stack([ExG, ExR, TGI], axis=2),
        ],
        axis=2,
    )
    descriptor_names = ["sR","sG","sB","H","S","V","L","a","b","L","u","v","Y","U","V","Y","Cb","Cr","ExG","ExR","TGI",]

    return descriptors


def classify_Necrosis(
    image: np.ndarray,
    model,
    contrasted: int = 0,
    mask: np.ndarray = None,) -> np.ndarray:
    """
    Be sure image is in BGR ! (to apply correctly contrast if needed)
    if binary veg/bckg mask is given: apply model only on vegetation pixels
    """

    # Reverse channels to have RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # If no mask, create one full of 1
    if mask is None:
        mask = np.ones((height, width))
    # If mask is given and there is no vegetation in mask: returns mask with only zeros
    if 1 not in np.unique(mask):
        return np.zeros(mask.shape)

    image = image.reshape((width * height), 1, 3)
    mask = mask.reshape((width*height), 1)
    yellow_green_mask = np.zeros(image.shape[:-1])
    vegetation_pixels = mask > 0
    image = image[None, vegetation_pixels]

    try:
        featured_image = get_features_Necrosis(image)
    except:
        # If full of Soil
        mask = np.zeros((height, width), dtype=np.uint8)
        return mask

    descriptors_flatten = featured_image.reshape(-1, featured_image.shape[-1])
    descriptors_flatten = xgb.DMatrix(descriptors_flatten)
    segmented_flatten_probs = model.predict(descriptors_flatten)
    y_pred = np.argmax(segmented_flatten_probs, axis=1)  # J0 V1 M2
    y_pred = [
        3 if x == 0 else x for x in y_pred
    ]  # Move J0 to J3 to let Soil as 0 ==> Final : 0 Soil | 1 Healthy | 2 Necrosis-Brown | 3 Chlorosis-Yellow

    yellow_green_mask[vegetation_pixels] = y_pred
    mask = yellow_green_mask.reshape((height, width))

    return mask


def visualisation_Necrosis(rgb_image: np.ndarray, yg_mask: np.ndarray, stg) -> np.ndarray:
    """
    take rgb image and yellow_green_mask and apply color
    """
    image_copy = rgb_image.copy()

    # Works because BGR
    if stg =="SegVeg_Necrosis":
      image_copy[yg_mask == 0] = (0, 0, 0)
      image_copy[yg_mask == 3] = (0, 204, 255)
      image_copy[yg_mask == 1] = (0, 100, 0)
      image_copy[yg_mask == 2] = (20, 61, 102)
    if stg =="SegVeg_XG_SVM":
      image_copy[yg_mask == 0] = (0, 0, 0)
      image_copy[yg_mask == 2] = (0, 204, 255)
      image_copy[yg_mask == 1] = (0, 100, 0)

    visualisation = cv2.addWeighted(rgb_image, 0.4, image_copy, 0.6, 0)

    index_class = np.unique(yg_mask)
    color_vizu = []
    id = [(0, 0, 0), (0, 100, 0), (20, 61, 102), (0, 204, 255)]
    for c in index_class:
        color_vizu.append(id[int(c)])

    return visualisation, index_class, color_vizu
```

# Predict and Visualize


```python
from PIL import Image, ImageCms
from skimage import color, io

list_images = ["/content/ESWW0003_20220701_soil1202205233_fake_1.png", "/content/ESWW0060033_20220617_BF0A9824_fake_2.png", "/content/ESWWpermanent4_20220613_soil_permanent2_20220521_fake_4.png"] #LiteralDataset_2

for i in list_images :

  image = cv2.imread(i)
  if image.shape[0] > 512 :
    image = cv2.resize(image, (512, 512))

  image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Reminder: Stage 1 is VegAnn, see previous part.
  # Overlay of stage 1 mask is then used in prediction_XG_SVM and classify_Necrosis functions
  preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')
  image_prcd = preprocess_input(image_RGB)
  image_prcd = image_prcd.astype('float32')

  inputs = torch.tensor(image_prcd)
  inputs = inputs.permute(2,0,1)
  inputs = inputs[None,:,:,:]
  logits = model(inputs)
  pr_mask = logits.sigmoid()

  pred = (pr_mask > 0.5).numpy().astype(np.uint8)
  pred = np.squeeze(pred)
  visualisation_image_Stg1, index_class, color_vizu = visualization(
      image_RGB, pred
  )
  visualisation_image_Stg1 = Text_Image(
    visualisation_image_Stg1,
    "VegAnn Whole Vegetation Segmentation",
    "VegAnn",
    index_class,
    color_vizu,
  )

  # SegVeg part #
  raw_yellow_green_mask_XG = prediction_XG_SVM(image, model_XG, mask=pred, threshold=0.5, contrasted=1)
  raw_visu_XG, index_class, color_vizu = visualisation_Necrosis(image, raw_yellow_green_mask_XG, "SegVeg_XG_SVM")
  raw_visu_XG = Text_Image(
    raw_visu_XG,
    "SegVeg XG SVM Segmentation",
    "SegVeg_XG_SVM",
    index_class,
    [(0, 0, 0), (0, 100, 0), (0, 204, 255)]
  )

  # Can be smoothed
  # yellow_green_mask = cv2.erode(raw_yellow_green_mask, np.ones((2,2), np.uint8), iterations=1)
  # yellow_green_mask = cv2.GaussianBlur(yellow_green_mask, (3,3), 0)

  yellow_green_mask_Necrosis = classify_Necrosis(image, model_Necrosis, mask=pred)
  visualisation_image_Stg3, index_class, color_vizu = visualisation_Necrosis(image, yellow_green_mask_Necrosis, "SegVeg_Necrosis")
  visualisation_image_Stg3 = Text_Image(
      visualisation_image_Stg3,
      "Necrosis Segmentation",
      "SegVeg_Necrosis",
      index_class,
      color_vizu)

  concatenated_image = np.concatenate((image_RGB, visualisation_image_Stg1, raw_visu_XG[..., ::-1], visualisation_image_Stg3[..., ::-1]), axis=1)
  # concatenated_image = cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2RGB)
  # cv2.imwrite("concatenated_output.jpg", concatenated_image)
  # break

  # Plotting the images
  fig, ax = plt.subplots(1, 1, figsize=(15, 7))
  plt.imshow(concatenated_image)
  # ax.set_title("Input Image + Prediction")
  plt.axis('off')  # Hide axes
  plt.show()

  print(str(i))
  print("succeeded ✅")

print("All images processed succeeded ✅")
print("END ✅")
```
![1_concatenated_output](https://github.com/user-attachments/assets/bf9769f4-0b47-4737-b125-659fec6e1262)

    /content/ESWW0003_20220701_soil1202205233_fake_1.png succeeded ✅
    
![2_concatenated_output](https://github.com/user-attachments/assets/359b7fb1-b129-4214-83d6-faa65b507e66)


    /content/ESWW0060033_20220617_BF0A9824_fake_2.png succeeded ✅

![3_concatenated_output](https://github.com/user-attachments/assets/e75f21fd-5b9f-4bf9-8aa2-7b91777befee)


    /content/ESWWpermanent4_20220613_soil_permanent2_20220521_fake_4.png succeeded ✅
    
    All images processed succeeded ✅
    END ✅



```python

```
