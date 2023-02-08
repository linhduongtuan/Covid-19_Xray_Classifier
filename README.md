# Covid-19_Xray_Classifier
You can find our paper here [https://www.medrxiv.org/content/10.1101/2020.08.13.20173997v1] & DOI [doi: https://doi.org/10.1101/2020.08.13.20173997].

At the moment, we are facing with one of the most crisis public health around the World, called Covid-19 pandemic. The F0 case was first reported in Wuhan, China. Now the pandemic threads all people lives as well as ecomoic collapse in most countries in the World.

To help physicians in the battles, I apply various deep learning to classify chest X-ray images from patients who are suspected to infect Covid-19.

I hope that the computer-aided tool can be robust, fast, and accurate diagnosis lungs images of Covid-19 infection, other pneumonia, and normal as well. 

Three models with transfer learning were trained with a public dataset of chest X-ray images to create classifiers. However, each trained model does not reach 100% accuracy so we apply ensemble voting method to increase both sensitivity and specificity for the overall classifier.

The first model used in the study is ResNet50 [https://arxiv.org/pdf/1512.03385.pdf]. Moreover, result indicates ResNet50 training with mage_size=512 can reach the best accuracy [found here: https://covidresearch.ai/datasets/dataset?id=2]

The second one is more recent state-of-the-art, EfficientNet_B0 with image size = 224 [https://arxiv.org/pdf/1905.11946.pdf] and code here [https://github.com/ufopcsilab/covid-19] or/and [https://github.com/ufopcsilab/EfficientNet-C19]. Some one should raise a question why I used EfficientNet_B0 instead of EfficientNet_L2? The answer can find here: [https://arxiv.org/pdf/2004.05717.pdf]. This paper shows the best result can be achieved using the simplest model in EfficientNet family.

Last but no least, I want to try novel SOTA model called TResNet_XL with image size=448. The architecture of TResNet family can be referenced paper [https://arxiv.org/pdf/2003.13630.pdf.]  and code here [https://github.com/mrT23/TResNet]  OR [https://github.com/rwightman/pytorch-image-models].

Pretrained weights of ResNet50 and EfficientNet, and many valuable scripts to train, validate, inference, and clean checkpoint are referenced at [https://github.com/rwighman/pytorch-image-models]. Thank @Ross Wightman for your great job!

## Dataset used :   
Dataset of chest X-Ray images was taken from the website: [https://covidresearch.ai/datasets/dataset?id=2]

The data could be collected from both papers  [Wang L, Wong A. COVID-net: A tailored deep convolutional neural network design for detection of COVID-19 cases from chest radiography images. arXiv:200309871 \[cs, eess\]. 2020. http://arxiv.org/abs/2003.09871. Accessed 10 Apr 2020.] AND [Cohen JP, Morrison P, Dao L. COVID-19 Image Data Collection. arXiv:200311597 \[cs, eess, q-bio\]. 2020. http://arxiv.org/abs/2003.11597. Accessed 11 Apr 2020.]

We also easily find codes for both papers here: [https://github.com/lindawangg/COVID-Net] AND [https://github.com/ieee8023/covid-chestxray-dataset]

## Summary my experiment's results

## Flow:
* To check my prediction results, you can use NoteBook folder [here] (https://github.com/linhduongtuan/Covid-19_Xray_Classifier/blob/master/Notebooks/Metrics.ipynb)
* Checkpoints of the models are loaded, found: [models.py](https://github.com/linhduongtuan/Covid-19-Xray-Classifier/blob/master/commons.py) 
* Inference of the model: [inference.py] (https://github.com/linhduongtuan/Covid-19_Xray_Classifier/blob/master/inference.py
* Run on local web: [app.py] (https://github.com/linhduongtuan/Covid-19_Xray_Classifier/blob/master/app.py) 
* My trained weights of the models can be downloaded here [https://github.com/linhduongtuan/Covid-19_Xray_Classifier/blob/master/releases/]

## Run on Ubuntu and MacOS, but not test on Windows - 
Make sure you have installed Python , Pytorch, Flask and other related packages, refer requirement.txt.

* _First download all the folders and files_     
`git clone https://github.com/linhduongtuan/Covid-19_Xray_Classifier.git`     
* _Then open the command prompt (or powershell) and change the directory to the path where all the files are located._       
`cd Covid-19_Xray_Classifier`      
* _Now run the following commands_ -        

`python app.py`     


This will firstly download the models and then start the local web server.

now go to the local server something like this - http://127.0.0.1:5000/ and see the result and explore.

##TODO
* NEED TO INTERNAL AND EXTERNAL VALIDITY
* Improve Specificity and Sensisitity of Covid-19 Chest Xray via ensemble voting
* Improve web interface and cybersecurity
* Enable to predict other formats of image such as DICOM, *png, *tiff,...
* Enable to predict a batch of images
...
### @creator - Duong Tuan Linh


## Citing

### BibTeX

```bibtex
@article{duong2022automatic,
  title={Automatic detection of Covid-19 from chest X-ray and lung computed tomography images using deep neural networks and transfer learning},
  author={Duong, Linh T and Nguyen, Phuong T and Iovino, Ludovico and Flammini, Michele},
  journal={Applied Soft Computing},
  pages={109851},
  year={2022},
  publisher={Elsevier}
  DOI={https://doi.org/10.1016/j.asoc.2022.109851}
}
```

### Latest DOI

[![DOI](https://zenodo.org/badge/168799526.svg)](https://doi.org/10.1016/j.asoc.2022.109851)
