# SegVeg

<div align="center">
	
![download (1)](https://github.com/mserouar/SegVeg/assets/57948061/82030dac-c7d9-4bf5-a06c-fc73ff0575fa)
	
**Python module for Senescent Vegetation Image Segmentation based on SVM/XGBoost.**
	

</div>


## 📚 ABSTRACT

SegVeg is a model for semantic segmentation of RGB images into background, green vegetation and senescent vegetation classes.
Link to original published paper : https://spj.sciencemag.org/journals/plantphenomics/2022/9803570/

### Useful information <a name="start"></a>

The method proposed may be described in two stages. 

In the first stage, the whole image is classified into Vegetation/Background mask using a U-net type Deep Learning network.
Then, the segmented vegetation pixels are classified into Green/Senescent vegetation using a binary SVM. 

Here, you will only find the Second stage (yellow part in Figure above).
To perform the first stage, please find more information on : https://github.com/simonMadec/VegAnn


## 📝 Citing

If you find this work useful in your research (Python module, model or Dataset), please cite both papers :

#### Paper Senescent Green Vegetation segmentation <a name="Paper"></a>

Serouart Mario Madec Simon David Etienne Velumani Kaaviya Lopez Lozano Raul Weiss Marie Baret Frédéric . SegVeg: Segmenting RGB Images into Green and Senescent Vegetation by Combining Deep and Shallow Methods. Plant Phenomics. 2022;2022:DOI:10.34133/2022/9803570

#### Paper Vegatation Background segmentation <a name="Paper"></a>

Madec, S., Irfan, K., Velumani, K. et al. VegAnn, Vegetation Annotation of multi-crop RGB images acquired under diverse conditions for segmentation. Sci Data 10, 302 (2023). https://doi.org/10.1038/s41597-023-02098-y



## ☸️ How to use

```Simply lauch the given .ipynb google collab in main directory :``` https://github.com/mserouar/SegVeg/blob/main/VegAnn_%2B_SegVeg.ipynb

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
Please follow below instructions if you want to know more (Docker, Bash command, Dev mode, Features, Supp. Materials, ...)
<br>

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

#### 2. Data + Model (XGBoost used by default in config file)

	__MODELS__

[[model_scikit](https://github.com/mserouar/SegVeg/blob/main/docs/DATA/model_scikit)] : Green/Senescent vegetation SVM model built with Scikit-learn CPU

[[XGBoost](https://github.com/mserouar/SegVeg/blob/main/docs/DATA/XGBoost)] : Green/Senescent vegetation XGBoost alternative model (for computational performances purposes)

	__LABELLED_PIXELS CSV__ 

[[LABELLED_PIXELS](https://github.com/mserouar/SegVeg/blob/main/docs/DATA/LABELLED_PIXELS.csv)] : Labelled pixels (Green Veg./Senescent Veg./Background) used to perform accuracy model and its train/test repartition information


	__RGB IMAGES AND MASKS__

[[DATASET](XXX)] : RGB images and Vegetation/Background masks from LITERAL, PHENOMOBILE AND LITERAL domains used to perform accuracy model

	__Ready-to-use__

[[Session 2021-03-17 14-19-59](https://github.com/mserouar/SegVeg/tree/main/docs/DATA/Session%202021-03-17%2014-19-59)] : Test Session 



## ☸️ How to use

#### 1. Launch the module (Once you launched the Standard installation section)


| Positionnal arguments    | Description           | 
| :------------- |:-------------|
| input_folder      | Directory of the session you want to process : **"PATH/FROM/GITHUB/docs/DATA/Session 2021-03-17 14-19-59/"** | 
| configuration_file       | Configuration file for hyperparameters tuning : **"PATH/FROM/GITHUB/config/yellowConfiguration.json"** | 
##

**/!\ Please modify PATH/FROM/GITHUB/ acording to your path in both shell args + in .json config file with model XGBoost path**  
##
```
EXAMPLE in shell :  yellowgreen-multi 'PATH/FROM/GITHUB/docs/DATA/Session 2021-03-17 14-19-59/''PATH/FROM/GITHUB/config/yellowConfiguration.json'
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
