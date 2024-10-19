import os
import operator
import cv2
import torch
from mmdet.apis import init_detector, inference_detector  # , show_result_pyplot
import mmcv
import numpy as np
import tifffile as tiff
import json
import time
import logging
# from D3_T import Height
# from Picture_T import Picture
import glob


DLdir = r'./dlconfig'
if not os.path.exists(DLdir):
    os.mkdir(DLdir)
if os.path.exists(os.path.join(DLdir, 'dlconfig.json')):
    with open(os.path.join(DLdir, 'dlconfig.json'), 'r') as f:
        dictout = json.load(f)
        issave_img = dictout['issave_img']  # true: save image
        img_path = dictout['img_path']
        checkpoint_file = dictout['checkpoint_file']
        f.close()
if not os.path.exists(img_path):
    os.mkdir(img_path)
# if not os.path.exists(img_path + '\\{}'.format(time.strftime("%Y-%m-%d"))):
#     os.mkdir(img_path + '\\{}'.format(time.strftime("%Y-%m-%d")))
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(filename)s %(levelname)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     filename=img_path + '\\{}.txt'.format(time.strftime("%Y-%m-%d"), time.strftime("%Y-%m-%d")),
#                     filemode='a')

config_file = r'model_file\rtmdet-ins_tiny_8xb32-300e_coco.py'

model = init_detector(config_file, checkpoint_file, device='cpu')
def inf(image):
    stock_retrigger = [[0,0,0,0],[0]]
    origin_shape = (image.shape[0], image.shape[1])  

    if image.shape[2] ==1:
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2RGB)
    result_detected = inference_detector(model, image)
    
    # print('result_detected',result_detected)
    resule_dict = result_detected.pred_instances.to_dict()
    bboxes = resule_dict['bboxes'].tolist()
    scores = resule_dict['scores'].tolist()
    labels = resule_dict['labels'].tolist()
    list_i = []
    for i in range(len(labels)):
        if scores[i] > 0.65:
            list_i.append(i)
        else:
            continue
        i += 1
    # print(list_i)
    bbox_label = []
    for j in range(len(list_i)):
        bbox_label.append([bboxes[j],labels[j]])
    print(bbox_label)
    if len(bbox_label) == 0:
        print('no can det')
        return stock_retrigger
    colors = [num *10 +10 for num in range(len(bbox_label))]
    print(colors)


    # bbox_label = [[x1,x2,y1,y2], label]
    # return result_detected

def test():
    dir_path = r"D:\testAI\mmdetection-TXG\image_test\image"
    path_list = glob.glob(dir_path + '/*.jpg')
    for img_path in path_list:
        image = cv2.imread(img_path)
        result = inf(image)
        # print('result',result)
        

test()