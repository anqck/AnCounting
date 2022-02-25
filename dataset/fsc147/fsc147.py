import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import json


class FSC147(Dataset):
    def __init__(self, data_path, transform=None, train='train',  patch=False, flip=False):
        self.data_path = data_path
        anno_file = data_path + 'annotation_FSC147_384.json'
        data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
        im_dir = data_path + 'images_384_VarV2'
        gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'

        with open(anno_file) as f:
            self.annotations = json.load(f)

        with open(data_split_file) as f:
            self.data_split = json.load(f)

        
        self.im_ids = self.data_split[train]
        
            
        self.img_map = {}
        self.img_list = []
        # loads the image/(examplar,gt) pairs
        for _, im_id in enumerate(self.im_ids):
            anno = self.annotations[im_id]
            bboxes = anno['box_examples_coordinates']

            rects = list()
            for bbox in bboxes[:3]:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            # if load_point:
            self.img_map['{}/{}'.format(im_dir, im_id)] = \
                                        {'lines_boxes':rects,\
                                          'density_map':gt_dir + '/' + im_id.split(".jpg")[0] + ".npy",\
                                          'points': anno['points']}
            # else:
            # img_map['{}/{}'.format(im_dir, im_id)] = \
            #                                 {'lines_boxes':rects,\
            #                                   'gt':gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"}
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]

        density_map_path = self.img_map[img_path]['density_map']
        lines_boxes = self.img_map[img_path]['lines_boxes']
        points = self.img_map[img_path]['points']
        
        # # load image and ground truth
        img, gt_density = load_data(img_path, density_map_path)


        sample = {'image':img,'lines_boxes':lines_boxes,'gt_density':gt_density}
        if self.transform is not None:
            sample = self.transform(sample)

        # print(sample['image'].shape, sample['boxes'].shape, sample['gt_density'].shape, len(points  ))

        return sample['image'], sample['boxes'],sample['gt_density'], points 
        # return img, (lines_boxes, density_map_gt,points)

def load_data(img_path, density_map_path):
    # load the images
    image = Image.open(img_path)
    image.load()

    #load density map
    density = np.load(density_map_path).astype('float32')    


    # # load ground truth points
    # points = []
    # with open(gt_path) as f_label:
    #     for line in f_label:
    #         x = float(line.strip().split(' ')[0])
    #         y = float(line.strip().split(' ')[1])
    #         points.append([x, y])

    return image, density