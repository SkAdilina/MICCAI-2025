import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

losses_used = "/exp_OURS"
task_path = "PATH_TO_TASK_DIR/Task_FreeSurfer_1-5T_longit"
task_name_exp = "Longit-OURS"
print("-------", task_name_exp, "--- in dataset.py----")
max_tp = 13

import pandas as pd
import fnmatch
import re



#vol_crop_shape = (112, 88, 88) 
vol_crop_shape = (96, 96, 96) 
print("CROP SIZE:", vol_crop_shape)

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class Pad(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        #image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        #label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        
        image = sample['image']
        label = sample['label']
        #print("beginning - label shape in tensor function", label.shape)
        total = image.shape[0]

        extra_shape = (1, 1) + vol_crop_shape
        #print(extra_shape)
        
        ## Making sure that all the numpy are same dimension of 12 time points
        #print("image shape before tensor", image.shape)
        for time in range(total, max_tp, 1):
            extra = np.full(extra_shape, 0)
            image = np.vstack([image, extra])
            label = np.vstack([label, extra])
            #print("image shape after stacking", image.shape)
        
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float()

        #print("after stacking - label shape in tensor function", label.shape)
        label = torch.from_numpy(label).long()
        label = label.squeeze(1) 
        label = F.one_hot(label, num_classes=3)
        #print("after one hot encoding - label shape in tensor function", label.shape)
        label = label.permute(0, 4, 1, 2, 3).contiguous()
        #label = label.squeeze() ## need to uncomment this for longitudinal dataset
        
        return {'image': image, 'label': label}


class more_Augmentations(object):

    def __call__(self, sample):

        image = sample['image']
        label = sample['label']
        #print("during augmentation", image.shape[0])

        if(random.uniform(0, 1)>0.5):
            #print("adding random noise..")
            rand_val =  random.uniform(0.1, 1)
            transform = tio.RandomNoise(mean=0, std=rand_val)
            for x in range(0, image.shape[0],1):
                image[x, :, :, :, :] = transform(image[x, :, :, :, :])
        
        if(random.uniform(0, 1)>0.5):
            #print("adding random gamma..")
            transform = tio.RandomGamma(log_gamma=(-0.3, 0.3))
            for x in range(0, image.shape[0],1):
                image[x, :, :, :, :] = transform(image[x, :, :, :, :])

        return {'image': image, 'label': label}

def cropping_function(volume):
    
    crop_size = vol_crop_shape
    image = volume
    
    #print("image shape in cropping_function beginning", image.shape)
    _, depth, height, width = image.shape
    
    # Calculate the starting indices for cropping
    start_depth = (depth - crop_size[0]) // 2
    start_height = (height - crop_size[1]) // 2
    start_width = (width - crop_size[2]) // 2

    # Calculate the ending indices for cropping
    end_depth = start_depth + crop_size[0]
    end_height = start_height + crop_size[1]
    end_width = start_width + crop_size[2]

    # Crop the volume
    image = image[:, start_depth:end_depth, start_height:end_height, start_width:end_width]

    return image

def transform(sample):
    trans = tio.Compose([
        more_Augmentations(),
        ToTensor()
    ])

    return trans(sample)


def transform_valid(sample):
    trans = tio.Compose([
        ToTensor()
    ])

    return trans(sample)

def string_to_tensor(s):
    # Convert string to list of ASCII values
    return torch.tensor([ord(c) for c in s], dtype=torch.int64)


def extract_timepoint(filename):
    """
    Sort by:
    1. Subject name
    2. Timepoint (bl=0, m06=6, m120=120, etc.)
    3. File type: regular < _ft < _seg
    """
    # Get base filename (remove path)
    fname = os.path.basename(filename).lower()

    # Extract subject name (everything before first "_")
    subject_match = re.match(r"([a-z0-9]+)", fname)
    subject_id = subject_match.group(1) if subject_match else ""

    # Extract timepoint
    if "bl" in fname:
        timepoint = 0
    else:
        match = re.search(r"m(\d{2,3})", fname)
        timepoint = int(match.group(1)) if match else float('inf')

    # Assign priority for file type: normal=0, _ft=1, _seg=2
    if "_seg" in fname:
        file_type = 2
    elif "_ft" in fname:
        file_type = 1
    else:
        file_type = 0

    return (subject_id, timepoint, file_type)


def extract_timepoint_only_seg(filename):
    """Extracts the month number from filenames. If 'bl' is present, returns 0."""
    if "bl" in filename.lower():
        return (0, "_seg" in filename)  # (time, is_seg)

    match = re.search(r"m(\d{2,3})", filename)
    timepoint = int(match.group(1)) if match else float('inf')
    
    return (timepoint, "_seg" in filename)  # Sort by time first, then _seg files last


class mydata_nnunet(Dataset):
    def __init__(self, pickle_root, mode='train', fold=0):
        self.mode = mode
        self.fold = fold
        self.paths = [os.path.join(pickle_root) + file for file in os.listdir(pickle_root)]

        file_path = pickle_root + self.mode + "_fold_" + str(self.fold) + ".txt"
        print(file_path)
        self.paths = []
        
        # Open the file in read mode
        with open(file_path, 'r') as file:
            content = file.read()
            lines = content.splitlines()
        #print(lines)

        # String to append in front of each item
        image_path = pickle_root + 'npy_all_normalized-96-96-96-Step1-GT-FT-crop-reg-nnunet-style/'
        #'npy_all_normalized-96-96-96-Step1-GT-crop-reg-nnunet-style/'
        #'npy_all_normalized-96-96-96-Step1-GT-crop-ONLY-nnunet-style/'
        #'npy_all_normalized-96-96-96-Step1-GT-crop-reg/'
        #'npy_all_normalized-96-96-96-Step1-GT/'
        #'npy_all_normalized/'
        #image_path = pickle_root + 'npy_all-partial/'
        
        for l in lines:
            #print(l)
            scan_list = [os.path.join(dirpath, fl)
                         for dirpath, dirnames, files in os.walk(image_path)
                         for fl in fnmatch.filter(files, l + '*')]
    
            #scan_list = np.sort(scan_list)
            # Sort scan_list based on the number after "m"
            scan_list = sorted(scan_list, key=extract_timepoint)
            #scan_list = [prefix + l + postfix for l in lines]
            #print(scan_list)
            self.paths.append(scan_list)

        #print(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        #for p in path:
        #    print(p)

        if self.mode == 'train':

            image = []
            label = []
            temp = []
            #getting the image
            #image = np.load(path[0])

            just_id = []
            cur_count = 1
            
            # getting the label
            label = np.load(path[2])
            #print(path[1])
            label = cropping_function(label)
            label [label == -1] = 0
            label = np.expand_dims(label, axis=0)

            #print("WHAT AM I GETTING FOR LABEL?")
            for time in range(5, len(path), 3):
                #print(path[time])
                temp = np.load(path[time])
                temp = cropping_function(temp)
                temp = np.expand_dims(temp, axis=0)
                #print("temp size", temp.shape)
                label = np.vstack([label, temp])
                #print("label shape after stacking", label.shape)
                cur_count = cur_count + 1

            #getting the rest of the time points
            #print("length of the path", len(path))
            just_name = path[0].split('/')[-1].replace(".npy", "")
            #print(just_name)

            just_id.append(just_name[0:-3])
            #print(just_id)
            
            image = np.load(path[0])
            image = cropping_function(image)
            image = np.expand_dims(image, axis=0)
            #print("image shape after cropping", image.shape)

            #print("WHAT AM I GETTING FOR IMAGE?")
            for time in range(3, len(path), 3):
                #just_name = path[time].split('/')[-1].replace(".npy", "")
                #print(just_name)
                #print(path[time])
                temp = np.load(path[time])
                temp = cropping_function(temp)
                temp = np.expand_dims(temp, axis=0)
                #print("temp size", temp.shape)
                image = np.vstack([image, temp])
                #print("image shape after stacking", image.shape)


            feat = np.load(path[1])
            #feat = np.expand_dims(feat, axis=0)
            #print("feature shape before loop", feat.shape)

            #print("WHAT AM I GETTING FOR FEATURE? IN TRAIN")
            for time in range(4, len(path), 3):
                #just_name = path[time].split('/')[-1].replace(".npy", "")
                #print(path[time])
                temp = np.load(path[time])
                #temp = np.expand_dims(temp, axis=0)
                #print("temp size", temp.shape)
                feat = np.vstack([feat, temp])
                #print("feature shape after stacking", feat.shape)
            for time in range(feat.shape[0], max_tp, 1):
                extra = np.full([1, feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]], 0)
                feat = np.vstack([feat, extra])
                #print("what is the shape of extra?", extra.shape)
                #print("feature shape after stacking extra", feat.shape)
                
            image = np.array(image)
            label = np.array(label)
            feat = np.array(feat)
            just_id = np.array(just_id)
            #print("shape of just_id", just_id.shape)
            
            #print("before", image.shape, label.shape)
            #print("image shape after all the stacking", image.shape)
            sample = {'image': image, 'label': label}
            sample = transform(sample)

            feat_tensor = torch.from_numpy(feat).float()
            
            #total_tp = int(len(path)/2)+1
            total_tp = cur_count

            #print("just_id type", type(just_id))
            
            return sample['image'].to(device), sample['label'].to(device), total_tp, just_id, feat_tensor.to(device)
            
            
        elif self.mode == 'valid' or self.mode == 'test':

            image = []
            label = []
            temp = []
            #getting the image
            #image = np.load(path[0])

            just_id = []
            cur_count = 1
            
            # getting the label
            label = np.load(path[2])
            #print(path[1])
            label = cropping_function(label)
            label [label == -1] = 0
            label = np.expand_dims(label, axis=0)
            
            for time in range(5, len(path), 3):
                temp = np.load(path[time])
                temp = cropping_function(temp)
                temp = np.expand_dims(temp, axis=0)
                #print("temp size", temp.shape)
                label = np.vstack([label, temp])
                #print("label shape after stacking", label.shape)
                cur_count = cur_count + 1

            #getting the rest of the time points
            #print("length of the path", len(path))
            just_name = path[0].split('/')[-1].replace(".npy", "")
            #print(just_name)

            just_id.append(just_name[0:-3])
            #print(just_id)

            #cur_count_img = 1
            image = np.load(path[0])
            image = cropping_function(image)
            image = np.expand_dims(image, axis=0)
            #print("image shape after cropping", image.shape)

            for time in range(3, len(path), 3):
                #just_name = path[time].split('/')[-1].replace(".npy", "")
                #print(just_name)
                temp = np.load(path[time])
                temp = cropping_function(temp)
                temp = np.expand_dims(temp, axis=0)
                #print("temp size", temp.shape)
                image = np.vstack([image, temp])
                #print("image shape after stacking", image.shape)
                #cur_count_img = cur_count_img + 1 

            feat = np.load(path[1])
            #feat = np.expand_dims(feat, axis=0)
            #print("image shape after cropping", image.shape)

            #print("WHAT AM I GETTING FOR FEATURE? IN VAL/TEST")
            for time in range(4, len(path), 3):
                #just_name = path[time].split('/')[-1].replace(".npy", "")
                #print(path[time])
                temp = np.load(path[time])
                #temp = np.expand_dims(temp, axis=0)
                #print("temp size", temp.shape)
                feat = np.vstack([feat, temp])
                #print("feature shape after stacking", feat.shape)
            for time in range(feat.shape[0], max_tp, 1):
                extra = np.full([1, feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4]], 0)
                feat = np.vstack([feat, extra])
                #print("what is the shape of extra?", extra.shape)
                #print("feature shape after stacking extra", feat.shape)

            
            image = np.array(image)
            label = np.array(label)
            feat = np.array(feat)
            just_id = np.array(just_id)
            #print("shape of just_id", just_id.shape)
            
            #print("before", image.shape, label.shape)
            #print("image shape after all the stacking", image.shape)
            sample = {'image': image, 'label': label}
            sample = transform_valid(sample)
            
            feat_tensor = torch.from_numpy(feat).float()
            
            #total_tp = int(len(path)/2)+1
            total_tp = cur_count
            #print("total tp count cur", cur_count)
            #print("cur count in img loop", cur_count_img)
            #print("just_id type", type(just_id))
            
            return sample['image'].to(device), sample['label'].to(device), total_tp, just_id, feat_tensor.to(device)
            
            
    def __len__(self):
        return len(self.paths)

    #def collate(self, batch):
    #    return [torch.cat(v) for v in zip(*batch)]
    
    def collate(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)  # Shape: (batch_size, time_steps, channels, height, width, depth)
        labels = torch.stack([item[1] for item in batch], dim=0)  # Same shape as images
        total_tps = torch.tensor([item[2] for item in batch], dtype=torch.int)  # Convert list of integers to a tensor
        ids = np.array([item[3] for item in batch])  # Keep IDs as a NumPy array
        ft_all = torch.stack([item[4] for item in batch], dim=0)  # Different shape from images
        
        return images, labels, total_tps, ids, ft_all






