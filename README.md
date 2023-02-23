# Summary

This project was launched as part of the 2022 [Brainhack Geneva](https://github.com/brainhack-ch/piriform-cortex-diversity). Our goal was to develop a pipeline to cluster neurons located in the piriform cortex by using their morphological features.
# Installation Guide
## Installing Miniconda
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a package management system for Python. It allows to create virtual environments containing different code packages for each project you're working on. Basically we will start by installing Miniconda, then we will create a virtual environment in which we will install all of the packages we need to run our code.  

The instructions to install miniconda can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/macos.html) **but** you can directly download the [installer for MacOS](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg). Follow the instructions and you'll get Miniconda on your computer.

All interactions with conda are done through the terminal, once you are done with the install, open a terminal window and type  
`conda info`  
and hit ENTER and a few lines should appear with informations about your conda installation and current environment.  

## Creating a virtual environment for the project
The first step is to create an environment that contains the necessary packages that our code depends on. Download the `environment.yml` file. Then from the terminal navigate to the folder that contains this file using for example :  
`cd /Users/hugofluhr/Downloads`  
(`cd` stands for "change directory")
then use the following command to create the environment :  
`conda env create -n piriform-cortex --file environment.yml`  
If prompted, enter `y` and ENTER to accept the installation of packages. Now you have created a new environment called `piriform-cortex` that contains multiple python packages that we need for the project.  

The environment should hopefully be set up and now anytime you want to work on the project you need to :
1) Open a Terminal window
2) Type `conda activate piriform-cortex` to activate the virtual environment and gain access to all the packages we have installed in it.

## Install the code we have

The easiest way to get the code from this repository is to click on "Code" (green button) and "download ZIP". Once the download is complete, unzip to the desired folder amd then open a terminal window and do the following :  
`cd path/to/the/code` with the appropriate path to the folder that you just downloaded.
`conda activate piriform-cortex` to activate the environment.  
`jupyter notebook` to launch a jupyter notebook server on your web browser from which you can open the _clustering_ notebook at `notebooks/clustering.ipynb`. Finally, you may follow the instructions on the notebook to run the piriform cortex neurons clustering.