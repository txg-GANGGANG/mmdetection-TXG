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
    if image.shape[2] ==1:
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2RGB)
    result_detected = inference_detector(model, image)
    # print('result_detected',result_detected)
    bbox_result, segm_result = result_detected
    print('bbox_result',bbox_result)
    print('segm_result',segm_result)
    
    resule_dict = result_detected.pred_instances.to_dict()
    bboxes = resule_dict['bboxes']
    scores = resule_dict['scores']
    labels = resule_dict['labels']
    print(type(labels))
    i = 0
    for score in scores:
        result_0 = []
        result_1 = []
        if score > 0.8:
            result_0.append(bboxes[i])
            result_1.append(labels[i])
            i += 1
        else:
            continue
        print('result_0',result_0)
        print('result_1',result_1)

    # print('result_list',result_list)
    # print('bboxes',bboxes['bboxes'])
    # bboxes = np.vstack(bbox_result)
    print('bboxes',bboxes)
    # return result_detected

def test():
    dir_path = r"D:\testAI\mmdetection-TXG\image_test\image"
    path_list = glob.glob(dir_path + '/*.jpg')
    for img_path in path_list:
        image = cv2.imread(img_path)
        result = inf(image)
        # print('result',result)
        

test()