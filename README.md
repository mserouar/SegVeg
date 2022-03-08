# SegVeg

<div align="center">

![logo](https://i.ibb.co/dc1XdhT/Segmentation-Models-V2-Side-1-1.png)
**Python module for Senescent Vegetation Image Segmentation based on SVM.**

</div>

## 📚 Abstract

ABSTRACT

### ⏳ Useful information <a name="start"></a>

The method proposed in [[paper](https://arxiv.org/abs/1505.04597)]  may be described in two stages. 

In the first stage, the whole image is classified into Vegetation/Background mask using a U-net type Deep Learning network.
Then, the segmented vegetation pixels are classified into Green/Senescent vegetation using a binary SVM. 

Here, you will only find the Second stage (yellow part in Figure above).
To perform the first stage, please find more information on : ⌚ **WORK IN PROGRESS** ⌚

## 📦 DATA <a name="models"></a>

All freely available DATA could be found in the [[docs](https://smp.readthedocs.io/en/latest/models.html#unet)] folder.

#### 1. Features


| Features      | Description           | 
| :------------- |:-------------|
| xx, yy      | Pixels position according to PIL Images ⚠️ Reversed in cv2 ⚠️. More information on : https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system | 
| R, G, B      | from RGB channels      | 
| H, S  | from HSL channels      |
| a, b | from CIELab channels     |
| GE  | Greyscale     |
| M, YE  | from CMYK channels      |
| Cb, Cr  | from YCbCr channels    |
| I, Q  | from YIQ channels     |
| Y  | Pixel labelled manually class, **0 for Senescence Veg., 1 for Green Veg., 2 for Soil/Backg.**   |

All colorspaces transformations are available in native script ```src/yellowgreenmulti/yellowgreenmultiutils.py``` or in the additionnal utils functions script ```src/yellowgreenmulti/util_fonctions.py```

#### 2. DATA

	__MODELS__

[[model_cuML](https://smp.readthedocs.io/en/latest/models.html#unet)] : Green/Senescent vegetation SVM model built with RAPIDs | cuML GPU in [[paper](https://arxiv.org/abs/1505.04597)]

[[model_scikit](https://smp.readthedocs.io/en/latest/models.html#unet)] : Green/Senescent vegetation SVM model built with Scikit-learn CPU in [[paper](https://arxiv.org/abs/1505.04597)]

	__PIXELS CSV__ 

[[VERY_ALL](https://smp.readthedocs.io/en/latest/models.html#unet)] : Whole annotated pixels used to perform accuracy model in [[paper](https://arxiv.org/abs/1505.04597)] 

[[USED](https://smp.readthedocs.io/en/latest/models.html#unet)] : Test pixels (Green/Senescent Veg.) used to perform accuracy model in [[paper](https://arxiv.org/abs/1505.04597)] and train/test repartition information


	__RGB IMAGES AND MASKS__

[[Literal](https://smp.readthedocs.io/en/latest/models.html#unet)] : RGB images and Vegetation/Background masks from LITERAL domain used to perform accuracy model in [[paper](https://arxiv.org/abs/1505.04597)]

[[PHENOMOBILE](https://smp.readthedocs.io/en/latest/models.html#unet)] : RGB images and Vegetation/Background masks from PHENOMOBILE domain used to perform accuracy model in [[paper](https://arxiv.org/abs/1505.04597)]

[[P2S2](https://smp.readthedocs.io/en/latest/models.html#unet)] : RGB images and Vegetation/Background masks from P2S2 domain used to perform accuracy model in [[paper](https://arxiv.org/abs/1505.04597)]


	__Ready-to-use__

[[Session 2021-03-17 14-19-59](https://smp.readthedocs.io/en/latest/models.html#unet)] : Test Session 

## 📝 Citing

If you find this work useful in your research, please cite either :

#### Python Module <a name="Module"></a>

```
@misc{SegVeg,
  Author = {Serouart Mario, Madec Simon},
  Title = {SegVeg},
  Year = {2022},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/mserouar/SegVeg}}
}
```
#### Images Dataset <a name="Images"></a>

```
@dataset{SegVeg,
  author    = {Serouart Mario, Madec Simon},
  title     = {Senescent Vegetation Dataset},
  year      = {2022},
  doi       = {XX.XXXX/XXX.XXXX.XXXXXXXX},
  Howpublished = {\url{https://ZENODO}}
  publisher = {Zenodo},
}
```

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

#### 1. Launch the module


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
| output      | Path in output_folder (l2) to save raw segmentation | 


	__model_parameters__

| Item    | Description           | 
| :------------- |:-------------|
| path_tofind      | Path to find the trained - Green and Yellow vegetation - model | 
| n_cores       | Number of cpu core used to predict pixels class ⚠️ Deprecated if you use the non parallelized but GPU based model_cuML (need to be installed accrding to : https://rapids.ai/start.html#rapids-release-selector) ⚠️ | 
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
