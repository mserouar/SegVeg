# SegVeg

<div align="center">
	
![logo](https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png)
**Python module for Senescent Vegetation Image Segmentation based on SVM.**
	
⚠️**SegVeg+ is available with better estimation, contact me to get the whole new model**⚠️
	
</div>


## 📚 ABSTRACT

The pixels segmentation of high resolution RGB images into background, green
vegetation and senescent vegetation classes is a first step often required before
estimating key traits of interest including the vegetation fraction, the green area index,
or to characterize the sanitary state of the crop.  We developed the SegVeg model for
semantic segmentation of RGB images into the three classes of interest. It is based on
a U-net model that separates the vegetation from the background. It was trained over a
very large and diverse dataset. The vegetation pixels are then classified using a SVM
shallow machine learning technique trained over pixels extracted from grids applied to
images.

Results show that the SegVeg model allows to segment accurately the three classes,
with however some confusion mainly between the background and the senescent
vegetation, particularly over the dark and bright parts of the images. The use of the
components of several color spaces allows to better classify the vegetation pixels into
green and senescent ones. Finally, the model is used to predict the fraction of the three
classes over the grids pixels or the whole images. Results show that the green fraction
is very well estimated (R²=0.94), while the senescent and background fractions show
slightly degraded performances (R²=0.70 and 0.73, respectively).

We made SegVeg publicly available as a ready-to-use script, as well as the entire
dataset, rendering segmentation accessible to a broad audience by requiring neither
manual annotation nor knowledge, or at least, a pre-trained model to more specific
use.

### ⏳ Useful information <a name="start"></a>

The method proposed in [[paper](XXX)]  may be described in two stages. 

In the first stage, the whole image is classified into Vegetation/Background mask using a U-net type Deep Learning network.
Then, the segmented vegetation pixels are classified into Green/Senescent vegetation using a binary SVM. 

Here, you will only find the Second stage (yellow part in Figure above).
To perform the first stage, please find more information on : ⌚ **WORK IN PROGRESS** ⌚

 🍎 **If you have any question, please open an issue**

## 📦 DATA <a name="models"></a>

#### 1. Features (If you would like to train a new model, if not just use given model on images through launching command)

| Features      | Description           | 
| :------------- |:-------------|
| Image_Name    | Name of corresponding images for each row pixel     | 
| DOMAIN  | From which sub-dataset explained in paper  | 
| REP | Repartition information for the trained given model  | 
| xx, yy      | Pixels position according to PIL Images ⚠️ Reversed in cv2 ⚠️. More information on : https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system | 
| R, G, B      | from RGB channels      | 
| Y  | Pixel labelled manually class, **0 for Senescence Veg., 1 for Green Veg., 2 for Soil/Backg.**   |

All colorspaces transformations are available in native script ```src/yellowgreenmulti/yellowgreenmultiutils.py```

#### 2. DATA (XGBoost used by default in config file)

	__MODELS__

[[model_scikit](https://github.com/mserouar/SegVeg/blob/main/docs/DATA/model_scikit)] : Green/Senescent vegetation SVM model built with Scikit-learn CPU

[[XGBoost](https://github.com/mserouar/SegVeg/blob/main/docs/DATA/XGBoost)] : Green/Senescent vegetation XGBoost alternative model (for computational performances purposes)

	__LABELLED_PIXELS CSV__ 

[[LABELLED_PIXELS](https://github.com/mserouar/SegVeg/blob/main/docs/DATA/LABELLED_PIXELS.csv)] : Labelled pixels (Green Veg./Senescent Veg./Background) used to perform accuracy model and its train/test repartition information


	__RGB IMAGES AND MASKS__

[[DATASET](https://smp.readthedocs.io/en/latest/models.html#unet)] : RGB images and Vegetation/Background masks from LITERAL, PHENOMOBILE AND LITERAL domains used to perform accuracy model

(Available in Zenodo platform following link)

	__Ready-to-use__

[[Session 2021-03-17 14-19-59](https://github.com/mserouar/SegVeg/tree/main/docs/DATA/Session%202021-03-17%2014-19-59)] : Test Session 

## 📝 Citing

If you find this work useful in your research (Python module, model or Dataset), please cite:

#### Paper <a name="Paper"></a>

```
% @article{SegVeg,
% 	title = {Title},
% 	shorttitle = {SegVeg},
% 	url = {XXXX},
% 	journal = {arXiv:2105.07660 [cs]},
% 	author = {},
% 	month = jun,
% 	year = {2021}
% }
```

## ☸️ How to use

#### 1. Launch the module (Once you launched the Standard installation section)


| Positionnal arguments    | Description           | 
| :------------- |:-------------|
| input_folder      | Directory of the session you want to process : **"PATH/FROM/GITHUB/docs/DATA/Session 2021-03-17 14-19-59/"** | 
| configuration_file       | Configuration file for hyperparameters tuning : **"PATH/FROM/GITHUB/config/yellowConfiguration.json"** | 

```
EXAMPLE in shell :  yellowgreen-multi 'PATH/FROM/GITHUB/docs/DATA/Session 2021-03-17 14-19-59/' 'PATH/FROM/GITHUB/config/yellowConfiguration.json'
```

#### 2. Understanding the config file

	__relative_paths__

| Item    | Description           | 
| :------------- |:-------------|
| rgb_images       | Path in input_folder (l1) where to find RGB images | 
| vegetation_masks       | Path in input_folder (l1) where to find binary whole -Green and Senescent- vegetation masks (0 and 1, instead of 0 and 255) | 
| visualisation       | Path in output_folder (l2) where to find overlay visualisation results | 
| log       | Folder to save log infos | 

	__model_parameters__

| Item    | Description           | 
| :------------- |:-------------|
| model     | Path to find the trained - Green and Yellow vegetation - model | 
| n_cores       | Number of cpu core used to predict pixels class ⚠️ Deprecated if you do not use Parallel Processing ⚠️ | 
| thresh       | Set the probability threshold of binary model to handle sensitivity | 
| contrasted       | If 1/True, automatic color enhancement is performed, in order to use whole color distribution of each image | 


## 🛠 Installation <a name="installation"></a>

We recommend installing this module into a dedicated Python virtual environment to avoid dependency
conflicts or polluting the global Python distribution.


### Standard (from source)

For this install to work, **you need to have the git command available in your terminal**.

First, install the version of the dependencies known to work with the ``requirements.txt`` file
with pip, then install the module from the local source with pip:

```shell
pip install -r requirements.txt
pip install .
```

### Checking the installation

Now, you should have a new command ``yellowgreen-multi`` available in your terminal
and in any directory when the corresponding vitual environment is activated. You can test it with
the following command to display the help:

```shell
yellowgreen-multi --help
```

## Installing for development

If you need to work on the module for maintenance or updates, always use a dedicated Python virtual
environment. First install the dependencies with pip, then install the module in development mode
with pip. As for the source installation, the git command must be available in your terminal.

```shell
pip install -r requirements.txt
pip install -e .[dev]
```

The ``[dev]`` part corresponds to the extra dependencies used during development. In this case, it
will also install [Pylint](https://pylint.readthedocs.io/en/latest/) for doing static code analysis.

Pylint can analyze the entire source code with the following command:

```shell
pylint src/yellowgreenmulti
```

In development mode, pip will install a reference to the source code instead of doing a full install.
This will allow to update the source code and directly see the modified behavior with the installed
``yellowgreen-multi`` command.

### Other kind of installation : From a wheel

First, it is recommended to install the version of the dependencies known to work with the
``requirements.txt`` file with pip:

```shell
pip install -r requirements.txt
```

Then, simply install the wheel with pip:

```shell
pip install yellowgreenmulti-X.Y.Z-py3-none-any.whl
```

## Building the Docker image

First, create a directory named ``wheels`` into the root directory. Place any needed private wheels
inside it before building the image.

This project contains a ``Makefile`` to build the Docker image on **Linux**:

```shell
make build
```

Once done, you should have a new Docker image called ``yellowgreen-multi`` that you can
directly use to run the module. For example:

```shell
docker run --rm yellowgreen-multi --help
```
