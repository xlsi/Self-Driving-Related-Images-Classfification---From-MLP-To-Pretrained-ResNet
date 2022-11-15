# Self-Driving-Related-Images-Classfification---From-MLP-To-Pretrained-ResNet
This image classification task uses deep learning models, from basic MLP to unpretrained ResNet to pretrained ResNet, which is an overall process covering data collection to model training.  


**Our Findings and Thoughts**
* MLP has a low accuracy. And it can be deduced that it is maybe because MLP does not have a translation or rotational invariance.
* Unpretrained ResNet has a higher accuracy limit than MLP. However, it takes too many epochs to train it without pretrained CNN layers. Within 10 epochs, the max training accuracy and the max test accuracy are no more than 70 %, which is relatively slow. We can see that unpretrained ResNet performs badly on a such small dataset, also it is time-consuming.
* The pretrained CNN model shows a very satisfying result. Within 10 epochs, both training accuracy and test accuracy reach more than 90% and are almost converged. It shows the power of transfer learning on such a small dataset in computer vision tasks.

I hope you find this helpful!

## Requirements
This task is using Flickr API to collect image data. So make sure you install the bellow package before you run the make_datasets.py, or you should run the below command first.
```bash
(base) pip install flickrapi
```

## How To Use
In a nutshell here's how to use this template, so for example assume you want to implement pretrained ResNet50 to train mnist, so you should do the following:  
- In ./src/data folder, define your config and run make_datasets.py, then images will be stored under ./data folder.  
- In ./src/features folder, run image_preprocessing.py, and you can get plots of images' size and area, then it will delete outlier images in the ./data folder.  
- In ./src/models folder, there are three train.py you can choose to run, and it will save the model in the ./models folder. Then, you can use predict_model.py to use your model on new images. In the current predict_model.py, pretrained ResNet is used for new images.

## Folder Structure
```bash
├─data
│  └─Flickr_scrape
│      ├─automobile
│      ├─bicycle
│      └─pedestrian
├─models
├─notebooks
└─src
    ├─data
    ├─features
    ├─models
    └─visualization
```