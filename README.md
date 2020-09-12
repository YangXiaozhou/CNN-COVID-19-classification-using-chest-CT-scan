**After the sharing, please leave feedback for me by creating a new issue! I'll use this information to help improve the content and delivery of the material.**

# CNN-covid19-prediction-using-lung-CT-scan
COVID-19 prediction based on lung CT scan using convolutional neural network

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hugobowne/deep-learning-from-scratch-pytorch/f61063c3ec3aca124fd90b6af604e8e4c7313604?urlpath=lab)

# Description
- End product showcase
    + Showcase dog and cat classification results
    + Showcase COVID-19 prediction results
- Introduce the building blocks of convolutional neural networks for image classification (Foundation)
    + DNN
        * Motivation
        * Building blocks
    + CNN
        * Motivation
        * Features
- Demostrate how to build a simple CNN image classifier using tensorflow.keras (Toy example)
    + Step 1:
    + Step 2:
    + Step 3:
    + Step 4:
- Solve the INFORMS QSR data challenge as a complete case study for image classification using CNN (Action!)
    + Step 1:
    + Step 2:
    + Step 3:
    + Step 4:


## Prerequisites

Audience will be expected to be comfortable using Python with some prior exposure to NumPy & Scikit-Learn. It would help if you knew

* programming fundamentals and the basics of the Python programming language (e.g., variables, for loops);
* a bit about `pandas` and DataFrames;
* a bit about Jupyter Notebooks;


**However, I think the most important and beneficial prerequisite is a will to learn new things so if you have this quality, you'll definitely get something out of this tutorial.**

If you are going to code along, have fun with the code and use the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3 (see below), please install it before the sharing.


## Getting set up computationally

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hugobowne/deep-learning-from-scratch-pytorch/master?urlpath=lab)

The first option is to click on the [Binder](https://mybinder.readthedocs.io/en/latest/) badge above. This will spin up the necessary computational environment for you so you can write and execute Python code from the comfort of your browser. This is a free service. Due to this, the resource is not guaranteed, though it usually works well. If you want as close to a guarantee as possible, follow the instructions below to set up your computational environment locally (that is, on your own computer).


### 1. Clone the repository

To get set up for this sharing, clone this repository. You can do so by executing the following in your terminal:

```
git clone https://github.com/YangXiaozhou/CNN-covid19-prediction-lung-CT-scan
```

Alternatively, you can download the zip file of the repository at the top of the main page of the repository. If you prefer not to use git or don't have experience with it, this a good option.

### 2. Download Anaconda (if you haven't already)

If you do not already have the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3, go get it (You can also do this without Anaconda: using `pip` to install the required packages, however Anaconda is great for Data Science and I encourage you to use it).

### 3. Create your conda environment for this sharing

Navigate to the relevant directory in your terminal`CNN-covid19-prediction-lung-CT-scan` and install required packages in a new conda environment:

```
conda env create -f environment.yml
```

This will create a new environment called CNN-covid19-prediction-lung-CT-scan. To activate the environment on OSX/Linux, execute

```
source activate CNN-covid19-prediction-lung-CT-scan
```
On Windows, execute

```
activate CNN-covid19-prediction-lung-CT-scan
```


### 4. Open your Jupyter notebook

In the terminal, execute `jupyter notebook`.

Then open the notebook `xxx.ipynb` and we're ready to get coding. Enjoy.


