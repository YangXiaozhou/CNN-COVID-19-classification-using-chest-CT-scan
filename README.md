**Please leave feedback for me by creating a new issue! I'll use this information to help improve the content and delivery of the material.**

# CNN-COVID-19-classification-using-chest-CT-scan
COVID-19 classification based on chest CT scan using convolutional neural network

# Description
- End product showcase
    + Showcase dog and cat classification results
    + Showcase COVID-19 classification results
- Introduce convolutional neural networks for image classification (Foundation)
    + DNN
        * Examples
        * Key components
    + CNN
        * Motivation
        * Key components
- Demostrate how to build a simple CNN image classifier using tensorflow.keras (Toy example)
- Solve the INFORMS QSR data challenge as a complete case study using CNN (Action!)


## Prerequisites

It would help if you knew

* programming fundamentals and the basics of the Python programming language (e.g., variables, for loops);
* a bit about Jupyter Notebooks;

**However, I think the most important and beneficial prerequisite is a will to learn new things so if you have this quality, you'll definitely get something out of this tutorial.**

If you are going to code along, have fun with the code and use the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3 (see below), please install it before the sharing.


## Getting set up computationally

### 1. Clone the repository

To get set up for this sharing, clone this repository. You can do so by executing the following in your terminal:

```
git clone https://github.com/YangXiaozhou/CNN-COVID-19-classification-using-chest-CT-scan
```

Alternatively, you can download the zip file of the repository at the top of the main page of the repository. If you prefer not to use git or don't have experience with it, this a good option.

### 2. Download Anaconda (if you haven't already)

If you do not already have the [Anaconda distribution](https://www.anaconda.com/download/) of Python 3, go get it (You can also do this without Anaconda: using `pip` to install the required packages, however Anaconda is great for Data Science and I encourage you to use it).

### 3. Create your conda environment for this sharing

Navigate to the relevant directory in your terminal`CNN-COVID-19-classification-using-chest-CT-scan` and install required packages in a new conda environment:

```
conda env create -f environment.yml
```

This will create a new environment called CNN-COVID-19-classification-using-chest-CT-scan. To activate the environment on OSX/Linux, execute

```
source activate CNN-COVID-19-classification-using-chest-CT-scan
```
On Windows, execute

```
activate CNN-COVID-19-classification-using-chest-CT-scan
```


### 4. Open your Jupyter notebook

In the terminal, execute `jupyter notebook`.

Then open the notebook `xxx.ipynb` and we're ready to get coding. Enjoy.


