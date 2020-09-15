**Please leave feedback for me by creating a new issue or through my email!**

# CNN-COVID-19-classification-using-chest-CT-scan
COVID-19 classification based on chest CT scan using convolutional neural network

# Description
There are two Jupyter notebooks in this repo (in `notebooks` folder). 

1. 1-Introduction-to-convolutional-neural-network
    - This notebook introduces deep neural network (DNN) and convolutional neural network (CNN) to those who are not familiar with this area.
    - I illustrate the key components in a DNN, motivation for CNN and features that make CNN powerful for image classification.
2.  2-COVID-19-classification-based-on-CT-scan
    - This notebook is a walk-through of a CNN COVID-19 CT scan classifier that we've built using `tensorflow.keras`.
    - We built the network as an entry to the INFORMS QSR [data challenge](https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=f404f7b8-fcd6-75d5-f7a7-d262eab132e7).

Team members who built the COVID classifier: A/P [Chen Nan](https://www.eng.nus.edu.sg/isem/staff/chen-nan/), [Shi Yuchen](https://www.linkedin.com/in/yuchen-shi-2830ba158/?originalSubdomain=sg), and me. 

## Prerequisites

It would help if you knew

* programming fundamentals and the basics of the Python programming language (e.g., variables, for loops);
* a bit about supervised learning problems (e.g., classification, train-test split);
* a bit about Jupyter Notebooks;

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

Then open the notebook `1-Introduction-to-convolutional-neural-network.ipynb` and you're ready to dive in. Enjoy.


