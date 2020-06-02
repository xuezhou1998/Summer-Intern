This is a tensorflow deep learning inception_v3 model customized for "Fundus Image Analysis for Diabetic Retinopathy and Macular edema Grading". 
The project is mainly a deep learning model that performs disease screening and diagnosis for diabetes by analyzing the images of human eye retina. The image data are downloaded from IDRiD: Diabetic Retinopathy, and are already converted into TFRecord data.
The code in this project is based on the open source inception_v3 model created by PanJinQuan in https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiC3O-RwOLpAhWlgnIEHSDvBzgQFjABegQICRAB&url=https%3A%2F%2Fgithub.com%2FPanJinquan%2Ftensorflow_models_learning%3Ffiles%3D1&usg=AOvVaw0u3kTPsnzqYme0soikuJSr.

# tensorflow_models_learning

## 1. Create TFRecord Data
The images are contained in the dataset directory. To create TFRecord, run create_tf_record.py</br>
> For InceptionNet V1:set resize_height and resize_width = 224 </br>
> For InceptionNet V3:set resize_height and resize_width = 299 </br>



## 2.Training

There are VGG, inception_v1, inception_v3, mobilenet_v and resnet_v1 documents for training. To begin with, first generate TFRecord data and start training.
> VGG training：vgg_train_val.py </br>

> inception_v1 training：inception_v1_train_val.py </br>
> inception_v3 training：inception_v3_train_val.py </br>
> mobilenet training：mobilenet_train_val.py </br>

## 3. Pre-trained Model
The inception_v3 model pre-trained with the image data downloaded from IDRiD, can be found at the model directory. 




