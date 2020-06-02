

# Inception_v3 Model for "Fundus Image Analysis for Diabetic Retinopathy and Macular edema Grading"

## This is a tensorflow deep learning inception_v3 model customized for "Fundus Image Analysis for Diabetic Retinopathy and Macular edema Grading". 

## The project is mainly a deep learning model that performs disease screening and diagnosis for diabetes by analyzing the images of human eye retina. The image data are downloaded from IDRiD: Diabetic Retinopathy, and are already converted into TFRecord data.

## The code in this project is based on the open source inception_v3 model created by PanJinQuan in https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiC3O-RwOLpAhWlgnIEHSDvBzgQFjABegQICRAB&url=https%3A%2F%2Fgithub.com%2FPanJinquan%2Ftensorflow_models_learning%3Ffiles%3D1&usg=AOvVaw0u3kTPsnzqYme0soikuJSr.

## 1. Create TFRecord Data
The images are contained in the dataset directory. To create TFRecord, run create_tf_record.py</br>

If you have your own images, you need to do the following process:
Images -> Run create_labels_files.py -> Run create_tf_record.py => TFRecord data

Organize your images data like this: 
train/YourLabels/Image files
val/YourLabels/Image files
records/train299.tfrecords(blank)
records/val299.tfrecords(blank)
train.txt(blank)
val.txt(blank)
labels.txt(You need to manually fill in the labels of your images, 1 label per line in a txt file)





## 2.Training

After you generated your TFRecord, run inception_v3_train_val.py


## 3. Pre-trained Model
The inception_v3 model pre-trained with the image data downloaded from IDRiD, can be found at the model directory. 




