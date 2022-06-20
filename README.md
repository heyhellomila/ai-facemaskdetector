# COMP 472: Course Project - AI Face Mask Detector 

*Project Guidelines provided by Dr. Rene Witte of Concordia University, Summer 2022.*

**AI Face Mask Detector** is a Python library to create an AI that can analyse face images and detect whether a person is wearing a face mask or not, as well as the type of mask that is being worn.

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h2vfads4prj20u00vudkd.jpg)

This project focuses developing a Deep Learning *Convolutional Neural Network (CNN)* using PyTorch and train it to recognize four different classes: 

​				*(1)* Person without a face mask,
​				*(2)* Person with a “community” (cloth) face mask,
​				*(3)* Person with a “surgical” (procedural) mask,
​				*(4)* Person with a “FFP2/N95/KN95”-type mask (you do not have to distinguish between them). 

The project does not consider other mask types (e.g., FFP3), face shields, full/half-face respirators, PPEs, or images that do not show a single face (e.g., groups of people).

You may use a GUI (such as Pycharm and Anaconda) as a comprehensive tool to install plugins and other packages.

### Python

It is recommended that you use Python 3.7 or greater, which can be installed either through the Anaconda package manager (see [below](https://pytorch.org/get-started/locally/#anaconda)), [Homebrew](https://brew.sh/), or the [Python website](https://www.python.org/downloads/mac-osx/).

## Environment





## Installation

As the file is too big for Github, we have provided a link to both dataset and cnn: https://drive.google.com/drive/folders/1pzzq1y0dVN09l_bRReLsX1l_GnNfxj4x

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Anaconda.

Packages required:

- TorchVision

```bash
conda install pytorch torchvision -c pytorch
```
- OpenCV

- If you would like to delegate the computation to your GPU instead install the following:
	- cudatoolkit=11.3
	
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```


## Setting up the code to your system

- First download/clone the cnn2_2.py file from our [github repo](https://github.com/heyhellomila/ai-facemaskdetector)

- Open it in your preferred IDE

Take note that due to the batch size of the testing set, the CNN is intended to run in a hybrid mode. The training is delegated to the GPU (cuda) and the testing to the CPU.

If you are running it on an online Jupyter Notebook, or you do not have a modern dedicated GPU (NVIDIA RTX 2000 series and newer) you should run it purely on the CPU.
```python
# If you would like to run it purely on your cpu uncomment line 25 
# device = 'cpu'
#to 
device = 'cpu'  

# and comment line 26
device=torch.device("cuda")
# To: 
# device=torch.device("cuda")
```

## Usage
Simply hit run on your IDE


## Contributing Members 
Role Assignments



Mila Roisin (29595774) 
Bozhidar Leshev (40105294) 
Deniz Dinchdonmez (40128249) 

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

