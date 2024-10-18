import numpy as np
import math
import tifffile as tiff
import logging
import halcon as ha
from halcon.numpy_interop import himage_as_numpy_array, himage_from_numpy_array
import time
import os
import json

DLdir = r'./dlconfig'
if os.path.exists(os.path.join(DLdir, 'dlconfig.json')):
    with open(os.path.join(DLdir, 'dlconfig.json'), 'r') as f:
        dictout = json.load(f)
        img_path = dictout['img_path']
        f.close()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=img_path + '\\{}\\{}.txt'.format(time.strftime("%Y-%m-%d"), time.strftime("%Y-%m-%d")),
                    filemode='a')


# global image_h
class Height(object):
    def __init__(self, h_image):
        self.list_h = []
        self.h_image = h_image
        # global image_h
        # image_h = h_image

    def himg_save(self, path, name):
        try:
            p = path + '/' + name + '.tiff'
            tiff.imsave(p, self.h_image)
        except:
            raise ValueError('tiff image save failed')

    def get_heihgt(self, point, diff=5):
        """from point to get mean of area, already considered nans, but noisy is not included
        Returns: return the list: yxz value in h_image for every single box
        """
        # *******dimension: y,x,z
        yxz_area = self.h_image[int(point[1]) - diff:int(point[1]) + diff, int(point[0]) - diff:int(point[0]) + diff]
        # print('yxzarea', yxz_area)
        # print(point)
        yxz = []
        yxz1 = ['y', 'x', 'z']
        for i in range(yxz_area.shape[-1]):
            try:
                single_area = yxz_area[:, :, i].flatten(0)
                nans = np.isnan(single_area)
                single_area = np.delete(single_area, np.where(nans))
                # txg 删除点云数据中所有值为0的数据
                single_area = np.delete(single_area, np.where(single_area == 0))
                # txg 获取区域内点云数据中位数
                # print('i:{}'.format(i), single_area)
                single_area_median = np.median(single_area)
            except:
                logging.info('center point cloud data error')
                # raise ValueError('single_area got error')
                continue
            if single_area_median is not None:
                yxz.append(single_area_median)
        # print('yxz',yxz)
        return yxz

    def getarea_H_halcon(self, mask_):
        # global image_h
        # print('get halcon')
        max_high_list = []
        try:
            maskimg = himage_from_numpy_array(mask_.astype('uint8'))
            maskregion = ha.threshold(maskimg, 1, 200)
            pointimg = himage_from_numpy_array(self.h_image)
            imageX, imageY, imageZ = ha.decompose3(pointimg)
            ObjectModel3DUpX = ha.reduce_domain(imageX, maskregion)
            ObjectModel3DUpY = ha.reduce_domain(imageY, maskregion)
            ObjectModel3DUpZ = ha.reduce_domain(imageZ, maskregion)
            ObjectModel3Ddown = ha.xyz_to_object_model_3d(ObjectModel3DUpX, ObjectModel3DUpY, ObjectModel3DUpZ)
            ObjectModel3DConnected = ha.connection_object_model_3d(ObjectModel3Ddown, 'distance_3d', 5)
            ObjectModel3DUp = ha.select_object_model_3d(ObjectModel3DConnected, 'num_points', 'and', 5000, 10000000)
            max_high = 660
            for i in range(len(ObjectModel3DUp)):
                GenParamValue = ha.get_object_model_3d_params(ObjectModel3DUp[i], 'point_coord_z')
                GenParamValue_s = np.delete(GenParamValue, np.where(GenParamValue == 0))
                if np.any(GenParamValue_s):
                    median = np.min(GenParamValue_s)
                    # print('max_high', median)
                    if max_high > median:
                        max_high = median
                else:
                    continue
            return max_high
        except:
            # print(11111111111111111)
            # print('getarea_H_halcon error')
            logging.info('getarea_H_halcon error')
            return 0
        
    def getheight_halcon(self, mask):
        """halcon-python get point_cloud value by maskregion and region"""
        maskimg = himage_from_numpy_array(mask.astype('uint8'))
        pointimg = himage_from_numpy_array(self.h_image)
        # logging.info('getted hal point image')
        imageX, imageY, imageZ = ha.decompose3(pointimg)
        maskregion = ha.threshold(maskimg, 1, 200)
        maskregion1 = ha.connection(maskregion)
        valueZ = ha.gray_features(maskregion1, imageZ, 'median')
        # print('--valueZ--', valueZ)
        min_, max_, range_ = ha.min_max_gray(maskregion1, imageZ, 25)
        # logging.info('valueZ:' + str(685 - valueZ[0]) + 'range_' + str(range_))
        return valueZ, range_
    
    
    
    def get_widhei_style(self, mask, box, center, list_result=None):
        global valueZ_s, style1
        if list_result is None:
            list_result = [0, 0, 0]
        list_xyz = [[0], [0], [0]]
        valuez = 0
        yxz_0 = self.get_heihgt([int(box[0][0]), int(box[0][1])], diff=5)
        yxz_1 = self.get_heihgt([int(box[1][0]), int(box[1][1])], diff=5)
        yxz_2 = self.get_heihgt([int(box[2][0]), int(box[2][1])], diff=5)
        yxz_3 = self.get_heihgt([int(box[3][0]), int(box[3][1])], diff=5)
        logging.info('goin halcon getheight_halcon')
        valueZ, range_ = self.getheight_halcon(mask)
        # print('valueZ', valueZ[0])
        # print('getvalue__')
        value_ = self.get_heihgt(center, diff=10)
        valueZ = [value_[2]]
        # if np.isnan(value_[2]).any():
        #     if np.isnan(valueZ[0]).any():
        #         valueZ = [valuez]
        # else:
        #     valueZ = [value_[2]]
        valueX = [value_[1]]
        valueY = [value_[0]]
        # print('valueY:', valueY[0])
        # print('valueX:', valueX)
        valueZ_s = 1010 - valueZ[0]
        if np.isnan(valueX[0]).any():
            logging.info("valueX is nan")
            # print("yxz_0 is nan")
            return list_result, valueZ, list_xyz
        if np.isnan(valueY[0]).any():
            logging.info("valueY is nan")
            # print("yxz_0 is nan")
            return list_result, valueZ, list_xyz
        # # yxz  ramge_ 为高低差，valz
        logging.info('valueZ:' + str(valueZ))
        # if valueZ_s >= 300 or valueZ_s <= 0:
        #     return list_result, valuez
        # print('range_', range_, valueZ)
        # print('valueZ_s___', valueZ_s)
        if np.isnan(yxz_0).any():
            logging.info("yxz_0 is nan")
            # print("yxz_0 is nan")
            return list_result, valueZ, list_xyz
        elif np.isnan(yxz_1).any():
            logging.info("yxz_1 is nan")
            # print("yxz_1 is nan")
            return list_result, valueZ, list_xyz
        elif np.isnan(yxz_2).any():
            logging.info("yxz_1 is nan")
            # print("yxz_2 is nan")
            return list_result, valueZ, list_xyz
        elif np.isnan(yxz_3).any():
            logging.info("yxz_3 is nan")
            # print("yxz_3 is nan")
            return list_result, valueZ, list_xyz
        # print('range' ,range_[0])
        # z0 = yxz_0[-1]
        # z1 = yxz_1[-1]
        # z2 = yxz_2[-1]
        # z3 = yxz_3[-1]
        # range_z = 660 - (max(z0, z1, z2, z3) - valueZ_s)
        # if range_z >= 30:
        #     logging.info("RECTANGLE_RANGE_Z > 30")
        #     print('range_z', range_z)
        #     return list_result, valueZ_s, list_xyz
        # if int(range_[0]) <= 30:
            # txg width,length判断吸盘策略
        width1 = int(math.sqrt(((yxz_0[0] - yxz_1[0]) ** 2) + (yxz_0[1] - yxz_1[1]) ** 2))
        length1 = int(math.sqrt(((yxz_0[0] - yxz_3[0]) ** 2) + (yxz_0[1] - yxz_3[1]) ** 2))
        length = max(width1, length1)
        width = min(width1, length1)
        # print("width0 : " + str(width) + "    length0 : " + str(length))
        logging.info("--width: " + str(width) + "--length: " + str(length))
        style1 = 1
        list_result = [length, width, style1]
        # print(valueZ[0])
        return list_result, valueZ_s, [valueX, valueY, valueZ]
