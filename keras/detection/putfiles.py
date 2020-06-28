import shutil

import csv
import os






filename_list=[]
files_labels_dict={}
labels_set=set()
with open('retinopathy_solution.csv', newline='') as csvfile:
     reader = csv.DictReader(csvfile)
     for row in reader:
        files_labels_dict[row['image']+'.jpeg']=row['level']
        labels_set.add(row['image']+'.jpeg')
#print(files_labels_dict)

for filename in os.walk('/Users/xuezhouwen/Desktop/Kaggle_Dataset/test'):
    if filename != ".DS_Store":
        # print("parent is: " + parent)
        # print("filename is: " + filename)
        # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
        
        filename_list.append(filename)
print(filename_list)
print(labels_set)
for parent, dirname, filenames in os.walk('/Users/xuezhouwen/Desktop/Kaggle_Dataset/test'):
    print("filename", filenames)
    for filename in filenames:
        
        if filename != ".DS_Store":
            if  filename in labels_set:
                original = '/Users/xuezhouwen/Desktop/Kaggle_Dataset/test/{}'.format(filename)
                target = '/Users/xuezhouwen/Desktop/Kaggle_Dataset/test1_dir/{}/{}'.format(str(files_labels_dict[filename]),filename)
                shutil.copyfile(original, target)
                print("copying {}", format(filename))

            
    



        
