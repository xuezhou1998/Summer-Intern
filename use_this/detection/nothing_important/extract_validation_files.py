





import shutil
import csv
import os



for parent, dirname, filenames in os.walk('./dataset/train1'):
    print("filename", filenames)
    extract_amount=int(0.2*len(filenames))
    for filename in filenames:
	    for parent_sub, dirname_sub, filenames_sub in os.walk(dirname):
		    for filename_sub in filenames_sub:
		        
		        if filename_sub != ".DS_Store":
		            if  filename_sub in labels_set:
		                original = '/Users/xuezhouwen/Desktop/Kaggle_Dataset/test/{}'.format(filename_sub)
		                target = '/Users/xuezhouwen/Desktop/Kaggle_Dataset/test1_dir/{}/{}'.format(str(files_labels_dict[filename_sub]),filename_sub)
		                shutil.copyfile(original, target)
		                print("copying {}", format(filename_sub))