import tarfile
import os, random
import shutil
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import nibabel
import numpy as np

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Alzheimer's Disease Classification")

parser.add_argument('--dataset_dir', type=str, default='./data', help='Directory for storing data_set')
parser.add_argument('--slice_ways', type=str, default='coronal', help='Select ways of slicing')

args = parser.parse_args()

## ===== ===== ===== ===== ===== ===== ===== =====
## Unzip the data_set
## ===== ===== ===== ===== ===== ===== ===== =====

raw_data_path = args.dataset_dir
dataset_path = os.path.join(raw_data_path, 'dataset')

tar_file_name = 'oasis_cross-sectional_disc'

for i in range(1, 13):
    tar = tarfile.open(os.path.join(raw_data_path, tar_file_name + str(i) + '.tar.gz'))
    tar.extractall(dataset_path)
    tar.close()
    print('Unzip operation completed：', tar_file_name + str(i) + '.tar.gz')

## ===== ===== ===== ===== ===== ===== ===== =====
## Move the data into one folder
## ===== ===== ===== ===== ===== ===== ===== =====

for i in range(1, 13):
    path = os.path.join(dataset_path, 'disc' + str(i))
    lst = os.listdir(path)
    for file in lst:
        shutil.move(os.path.join(path,file), dataset_path)

# remove the empty folder
for i in range(1, 13):
    os.rmdir(os.path.join(dataset_path ,'disc' + str(i)))

## ===== ===== ===== ===== ===== ===== ===== =====
## Take out the processed data
## ===== ===== ===== ===== ===== ===== ===== =====

lst = os.listdir(dataset_path)
processed_path = os.path.join(dataset_path,'processed')
if not os.path.exists(processed_path):
    os.mkdir(processed_path)

for filename in lst:
    if filename == 'processed':
        continue
    processed_path = os.path.join(dataset_path,filename,'PROCESSED','MPRAGE','T88_111')
    files = os.listdir(processed_path)
    for data_file in files:
        if data_file[-18:] == 't88_masked_gfc.hdr':
            shutil.copy(os.path.join(processed_path, data_file), os.path.join(dataset_path,'processed'))
        if data_file[-18:] == 't88_masked_gfc.img':
            shutil.copy(os.path.join(processed_path, data_file), os.path.join(dataset_path,'processed'))

## ===== ===== ===== ===== ===== ===== ===== =====
## Dataset split
## ===== ===== ===== ===== ===== ===== ===== =====

sum_lst = os.listdir(os.path.join(dataset_path))
sum_lst.remove('processed')
sum_lst.remove('val')
sum_lst.remove('train')
sum_lst.remove('test')

if not os.path.exists(os.path.join(dataset_path,'train')):
    os.mkdir(os.path.join(dataset_path,'train'))
if not os.path.exists(os.path.join(dataset_path,'val')):
    os.mkdir(os.path.join(dataset_path,'val'))
if not os.path.exists(os.path.join(dataset_path,'test')):
    os.mkdir(os.path.join(dataset_path,'test'))

# shuffle the data file list
random.shuffle(sum_lst)

df = pd.read_csv(os.path.join(dataset_path, 'oasis_cross-sectional.csv'))

y = list()
X = list()

for i in range(len(sum_lst)):
    # find corresponding label
    level = df.iloc[df[(df.ID == sum_lst[i])].index.tolist()[0],7]
    if level == 0.5 or level == 1 or level == 2: # CDR=0.5, 1, 2 -> label 1
        y.append(1)
        X.append(sum_lst[i])
    else: # CDR=0 -> label 0
        y.append(0)
        X.append(sum_lst[i])

# use stratify to ensure the proportion of labels in the training set, validation set and test set is the same
# split the data into 0.8(train), 0.2(test+validation)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
# split the 0.2 part into 0.1(test), 0.1(validation)
test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42,stratify=test_y)

# print the results for checking
pos_train = 0
pos_val = 0
neg_train = 0
neg_val = 0
for i in range(len(train_y)):
    if train_y[i] == 1:
        pos_train += 1
    if train_y[i] == 0:
        neg_train += 1
for i in range(len(val_y)):
    if val_y[i] == 1:
        pos_val += 1
    if val_y[i] == 0:
        neg_val += 1
print("pos_train",pos_train/len(train_y))
print("pos_val",pos_val/len(val_y))
print("neg_train",neg_train/len(train_y))
print("neg_val",neg_val/len(val_y))

## ===== ===== ===== ===== ===== ===== ===== =====
## Generate the data slices
## ===== ===== ===== ===== ===== ===== ===== =====

file_lst = os.listdir(processed_path)
if args.slice_ways == 'transverse':
    for file in file_lst:
        if file[-3:] == 'hdr':
            data = nibabel.load(os.path.join(processed_path, file)).get_fdata()
            data = np.rot90(data.squeeze(), 1)  # 208 * 176 * 176
            data = data[99:109, :, :]
            if file[:13] in train_x or file[:13] in val_x or file[:13] in test_x:
                if file[:13] in train_x:
                    folder = 'train'
                elif file[:13] in val_x:
                    folder = 'val'
                elif file[:13] in test_x:
                    folder = 'test'
                data = data.astype(np.uint16)
                new_data = np.zeros((10, 224, 224), dtype=np.uint16)
                new_data[:, 24:200, 24:200] = data
                for i in range(10):
                    img = Image.fromarray(new_data[i, :, :])
                    img.save(os.path.join(dataset_path, folder, file[:13] + '_' + str(i) + '.png'))
            else:
                continue

for file in file_lst:
    if file[-3:] == 'hdr':
        data = nibabel.load(os.path.join(processed_path,file)).get_fdata()
        data = np.rot90(data.squeeze(), 1) # 208 * 176 * 176
        if file[:13] in train_x or file[:13] in val_x or file[:13] in test_x:
            if file[:13] in train_x:
                folder = 'train'
            elif file[:13] in val_x:
                folder = 'val'
            elif file[:13] in test_x:
                folder = 'test'
            if args.slice_ways == 'transverse':
                data = data[:,:,83:93]
                data = data.astype(np.uint16)
                new_data = np.zeros((224, 224, 10), dtype=np.uint16) # resize the data into 224 * 224
                new_data[8:216, 24:200, :] = data
                for i in range(10):
                    img = Image.fromarray(new_data[:, :, i])
                    img.save(os.path.join(dataset_path, folder, file[:13] + '_' + str(i) + '.png'))
            elif args.slice_ways == 'sagittal':
                data = data[:, 83:93, :]
                data = data.astype(np.uint16)
                new_data = np.zeros((224, 10, 224), dtype=np.uint16)
                new_data[8:216, :, 24:200] = data
                for i in range(10):
                    img = Image.fromarray(new_data[:, i, :])
                    img.save(os.path.join(dataset_path, folder, file[:13] + '_' + str(i) + '.png'))
            else:
                data = data[99:109, :, :]
                data = data.astype(np.uint16)
                new_data = np.zeros((10, 224, 224), dtype=np.uint16)
                new_data[:,24:200,24:200] = data
                for i in range(10):
                    img = Image.fromarray(new_data[i, :, :])
                    img.save(os.path.join(dataset_path, folder, file[:13] + '_' + str(i) + '.png'))
        else:
            continue
