import cv2
import numpy as np
import logging
import halcon as ha
from halcon import himage_from_numpy_array
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


class Picture(object):
    # picture channel require channel equal 1 or 3 not 2
    def __init__(self, picture, pic_name, path):
        self.pic_name = pic_name
        self.path = path
        self.picture = picture
        # self.img_zeros = np.zeros(self.picture.shape)

    def picture_save(self, add_name, picture):
        logging.info('pic_save' + add_name + str(picture.shape))
        try:
            if len(picture.shape) == 2:
                picture = cv2.cvtColor(picture.astype('uint8'), cv2.COLOR_GRAY2RGB)
            if len(picture.shape) == 3:
                picture = picture.astype('uint8')
            cv2.imwrite(self.path + '/' + self.pic_name + add_name + '.jpg', picture)
        except:
            logging.info('picture_save function error')
            raise ValueError('channel= 2, error, picture channel require 3 ')
        return

    def draw_rotate_box(self, box, index):
        # print('box', box)
        self.picture = cv2.drawContours(self.picture, [box[9]], 0, (1, 124, 222), 2)
        # print('box[9]', [box[9]])
        # print(int(box[-1]))
        self.picture = cv2.putText(self.picture, str(int(box[-1])), (int(box[-4][0][0]), int(box[-4][0][1] - 10)),
                                   cv2.FONT_HERSHEY_COMPLEX, 0.6, (125, 65, 200), 1)
        self.picture = cv2.putText(self.picture, str(index), (int(box[-4][0][0]), int(box[-4][0][1] + 10)),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                   0.5, (0, 255, 0), 0)
        # self.picture = cv2.rectangle(self.picture, (int(box[-4][0][0]-10), int(box[-4][0][1]-10)), (int(box[-4][0][0]+10), int(box[-4][0][1]+10)),
        #                              (125, 65, 200), -1)
        return self.picture

    def draw_bbox(self, bboxes, index, hight_flag=False):
        self.picture = cv2.rectangle(self.picture, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])),
                                     (125, 65, 200), 1)
        self.picture = cv2.putText(self.picture, str(index), (int(bboxes[0]), int(bboxes[1])), cv2.FONT_HERSHEY_COMPLEX,
                                   0.5, (125, 65, 200), 1)
        if hight_flag:
            """if flag=true put the coordinates(xyz) on every bbox"""
            try:
                # print(bboxes)###to ensure the exact value in bboxes
                txt = 'z' + str(bboxes[-1]) + 'style' + str(bboxes[-4][-1])
                # logging.info(txt)
                self.picture = cv2.putText(self.picture, txt, (int(bboxes[0]), int(bboxes[1]) + 10 + index * 50),
                                           cv2.FONT_HERSHEY_COMPLEX, 0.6, (125, 65, 200), 1)
            except:
                logging.info('draw bbox function error')
                raise ValueError('bboxes value error')
        return self.picture

    def draw_segs(self, mask):
        """
        :param mask: single mask determined by segms output, the shape should be same as self.picture[0:2]
        :return: None, draw on picture the single contour which is target to pick, added with contour points and angle.
        """
        # logging.info('mask_shape:' + str(mask.shape))
        if mask.shape != self.picture.shape[0:2]:
            raise ValueError('zeros is not same')
        # get thresh in segs(self.picture)
        ret, thresh = cv2.threshold(mask.astype('uint8'), 0, 255,
                                    cv2.THRESH_BINARY)  ###cv2.threshold (源图片, 阈值, 填充色, 阈值类型)大于阈值置填充色，小于等于阈值置零
        contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.INTER_LINEAR)  ####contours得出mask目标一系列的坐标点
        if len(contours) == 0:
            logging.info('contours is none')
            return
        tarcontour = 0
        temsize = cv2.contourArea(contours[tarcontour])
        if len(contours) > 1:
            for ci in range(len(contours)):
                if temsize < cv2.contourArea(contours[ci]):
                    temsize = cv2.contourArea(contours[ci])
                    tarcontour = ci
        if temsize > 0:
            rect = cv2.minAreaRect(contours[tarcontour])
            # print("rect"+ str(rect))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # box = box.tolist()
            w_l_point = []
            # angle_d = rect[2]

            if rect[1][0] >= rect[1][1] and rect[2] <= 0:
                angle = abs(rect[2])
            else:
                angle = -(rect[2] + 90)
            area1 = cv2.contourArea(contours[tarcontour])
            # logging.info('segms_area:' + str(area1))
            # logging.info('angle:  ' + str(angle))

            return angle, box, rect, area1  # ,angle1

    def catch_rect(self, angle, center, style):
        # 计算正方形顶点的原始位置（未旋转）
        global c_x, c_y
        center = (center[0], center[1])
        # 获取旋转矩阵（2x2矩阵，不包含平移）
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        side_length = 175 / 1.4
        half_side = side_length / 2
        offset_value = 25 / 1.4
        # 0,1,3,6中心点不变
        if style == 2:
            c_x = center[0] + offset_value
            c_y = center[1]
            center = [c_x, c_y]
        if style == 4:
            c_x = center[0] + offset_value
            c_y = center[1] - offset_value
            center = [c_x, c_y]
        if style == 5:
            c_x = center[0]
            c_y = center[1] - offset_value
            center = [c_x, c_y]
        # 抓手外框
        # 1-5,2-56,3-456,4-2356,5-123456,
        points = np.array([
            [center[0] - half_side, center[1] - half_side],
            [center[0] + half_side, center[1] - half_side],
            [center[0] + half_side, center[1] + half_side],
            [center[0] - half_side, center[1] + half_side]
        ], dtype=np.float32)
        # 获取旋转矩阵（注意OpenCV中的角度是逆时针方向，所以可能需要转换角度方向）
        # 将点集重塑为(num_points, 1, 2)的形状
        points = points.reshape(-1, 1, 2)
        # print('points', points)
        center = (center[0], center[1])
        # 应用旋转矩阵到每个顶点
        rotated_points = cv2.transform(points, rotation_matrix)
        # print('--avoid_rotated_points', rotated_points)
        rotated_center_list = []
        # 1-5,2-56,3-456,4-2356,5-123456,策略
        center_list = np.array([
            [center[0] + 2 * offset_value, center[1] + 2 * offset_value], [center[0], center[1] + 2 * offset_value],
            [center[0] - 2 * offset_value, center[1] + 2 * offset_value], [center[0] + 2 * offset_value, center[1]],
            [center[0], center[1]],
            [center[0] - 2 * offset_value, center[1]],
            [center[0] + 2 * offset_value, center[1] - 2 * offset_value], [center[0], center[1] - 2 * offset_value],
            [center[0] - 2 * offset_value, center[1] - 2 * offset_value]],
            dtype=np.float32)
        # print('center_list',center_list)
        center_list = center_list.reshape(-1, 1, 2)
        rotated_catch_center = cv2.transform(center_list, rotation_matrix)
        rotated_center_list.extend(rotated_catch_center)
        # print('--rotated_catch_center--', rotated_catch_center)
        return points, rotated_points, rotated_center_list

    def draw_picker(self, rotated_points, rotated_catch_list, style):
        # print('rotate:::::::', rotated_catch_list, len(rotated_catch_list))
        cv2.drawContours(self.picture, [np.int32(rotated_points)], 0,
                         (0, 255, 0), 2)
        color = (0, 255, 0)
        # 1-5,2-56,3-456,4-2356,5-123456,
        for i, catch_center in enumerate(rotated_catch_list):
            # 提取或转换 catch_center 为合适的坐标形式
            catch_center = (catch_center[0][0], catch_center[0][1])
            # 设置默认颜色
            circle_color = color
            # 根据 style 修改特定圆圈的颜色
            if style == 0:
                continue
            if style == 1 and i == 4:  # 第5个圈（索引为4）颜色设为红色
                circle_color = (0, 0, 255)  # 红色，假设color格式为BGR
            elif style == 2:
                if i == 4 or i == 5:  # 第5个和第6个圈颜色设为红色
                    circle_color = (0, 0, 255)  # 红色
            elif style == 3:
                if i == 3 or i == 4 or i == 5:
                    circle_color = (0, 0, 255)  # 红色
            elif style == 4:
                if i == 1 or i == 2 or i == 4 or i == 5:
                    circle_color = (0, 0, 255)  # 红色
            elif style == 5:
                if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5:
                    circle_color = (0, 0, 255)  # 红色
            elif style == 6:
                circle_color = (0, 0, 255)  # 红色
            # 在图像上绘制圆圈
            cv2.circle(img=self.picture, center=tuple(catch_center), radius=int(25 / 1.4),
                       color=circle_color, thickness=2)
            # logging.info('CATCH_POINT HAD DRAW')
    
    def put_center_2d(self,center,angle,rect,style_put):#2d
        logging.info('rect_hw--'+str(rect[1][1])+str(rect[1][0]))
        logging.info('2d_rect angle is '+str(rect[-1]))
        # angle = abs(float(rect[-1]))
        # angle = float(rect[-1])
        if rect[1][0] >= rect[1][1] and rect[2] <= 0:
            angle = rect[2]+90
        else:
            angle = abs(rect[2])
        logging.info('BOX angle '+str(angle))
        logging.info('put begain')
        # print('put center angle '+str(angle))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)# 获取旋转矩阵
        h1 = rect[1][1]
        w1 = rect[1][0]
        h = max(h1,w1)
        w = min(h1,w1)
        logging.info('BOX --H--W--'+str(h)+'---'+str(w))
        # h = 580
        # w = 380
        # 获取BOX的像素长宽
        points =np.array([
            [center[0] , center[1]]], dtype=np.float32)
        if style_put == 1:
            # 分为2*1两个格子
            half_h = h/4
            points=np.array([
            [center[0] , center[1] - half_h],
            [center[0] , center[1] + half_h]], dtype=np.float32)
        if style_put == 2:
            # 分为2*2个格子
            half_h = h/4
            half_w = w/4
            points=np.array([
            [center[0] - half_w , center[1] - half_h],
            # [center[0] - half_w , center[1] + half_h],
            [center[0] + half_w , center[1] - half_h],
            [center[0] + half_w , center[1] + half_h],
            ],dtype=np.float32)            
        if style_put == 3:
            # 分为3*2个格子
            half_h = h/6
            half_w = w/4
            points=np.array([
            [center[0] - half_w , center[1] - 2*half_h],
            [center[0] + half_w , center[1] - 2*half_h],
            # [center[0] - half_w , center[1]],
            # [center[0] + half_w , center[1]],
            # [center[0] - half_w , center[1] + 2*half_h],
            # [center[0] + half_w , center[1] + 2*half_h],
            ],dtype=np.float32)
        # 将点集重塑为(num_points, 1, 2)的形状
        if style_put ==4:
            half_h = h/8
            half_w = w/4
            points=np.array([
                [center[0] - half_w, center[1]-3*half_h],
                [center[0] + half_w, center[1]-3*half_h],
            ],dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        # print('points', points)
        center = (center[0], center[1])
        # 应用旋转矩阵到每个顶点
        rotated_points = cv2.transform(points, rotation_matrix)
        # rotated_points.extend(rotated_points)
        # for put_center in rotated_points:
        #     cv2.circle(self.picture, (int(put_center[0][0]), int(put_center[0][1])),
        #                         5, (0,255,0), -1)
        # print('rotated_points', rotated_points)
        return rotated_points

    def put_center_3d(self,center,angle,rect,hws_list,style_put):#3d
        angle = abs(float(rect[-1]))
        logging.info('put_center_3d ')
        # angle = float(rect[-1])
        logging.info('put center angle '+str(angle))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)# 获取旋转矩阵
        h = hws_list[0]
        w = hws_list[1]
        # h = 580
        # w = 380
        # 获取BOX的像素长宽
        points =np.array([
            [center[0] , center[1]]], dtype=np.float32)
        if style_put == 1:
            # 分为2*1两个格子
            half_h = int(h/4)
            # half_w = int(w/2)
            points=np.array([
            [center[0]- half_h , center[1]]],
            # [center[0]+ half_h , center[1]]],
              dtype=np.float32)
        if style_put == 2:
            # 分为2*2个格子
            half_h = int(h/4)
            half_w = int(w/4)
            points=np.array([
            [center[0] - half_h , center[1] - half_w],
            [center[0] - half_h , center[1] + half_w],
            # [center[0] + half_h , center[1] - half_w],
            # [center[0] + half_h , center[1] + half_w],
            ],dtype=np.float32)            
        if style_put == 3:
            # 分为3*2个格子
            half_h = h/6
            half_w = w/4
            points=np.array([
            [center[0] - half_h , center[1] - 2*half_w],
            [center[0] + half_h , center[1] - 2*half_w],
            [center[0] - half_h , center[1]],
            [center[0] + half_h , center[1]],
            [center[0] - half_h , center[1] + 2*half_w],
            [center[0] + half_h , center[1] + 2*half_w],
            ],dtype=np.float32)
        if style_put == 4:
            # 分为4*2个格子
            half_h = h/8
            half_w = w/4
            points=np.array([
                [center[0] - 3*half_h , center[1] - half_w],
                [center[0] - 3*half_h , center[1] + half_w],
                [center[0] - half_h , center[1] - half_w],
                [center[0] - half_h , center[1] + half_w],
                [center[0] + 3*half_h , center[1] - half_w],
                [center[0] + 3*half_h , center[1] - half_w],
                [center[0] + half_h , center[1] - half_w],
                [center[0] + half_h , center[1] + half_w],
            ],dtype=np.float32)
        # 将点集重塑为(num_points, 1, 2)的形状
        points = points.reshape(-1, 1, 2)
        # print('points', points)
        center = (center[0], center[1])
        # 应用旋转矩阵到每个顶点
        rotated_points = cv2.transform(points, rotation_matrix)
        # rotated_points.extend(rotated_points)
        # for put_center in rotated_points:
        #     cv2.circle(self.picture, (int(put_center[0][0]), int(put_center[0][1])),
        #                         5, (0,255,0), -1)
        # print('rotated_points', rotated_points)
        return rotated_points

    def obstacle_avoidence(self, listdata):
        # img1 = self.picture.copy()
        if len(listdata) >= 1:

            # heights = listdata[-1]
            # print('height', heights)
            for index, box_value in enumerate(listdata):
                obstacle_flag = True
                # print('box_value', box_value)
                # self.draw_rotate_box(box=box_value, index=index)
                # txg 策略偏移后，避障优化
                angle = box_value[8]
                style = box_value[-5][-1]
                center = box_value[-3]
                height_index = int(box_value[-1])
                picker_points, rotated_points, rotated_catch_list = self.catch_rect(angle=angle, center=center,
                                                                                    style=style)
                listdata[index][7] = np.int0(picker_points)
                rect1 = cv2.minAreaRect(rotated_points)
                # print('listdata[index][7]',listdata[index][7])
                
                rect1_mask = box_value[-4]
                # txg 长或宽>330并且高度大于210
                # print('box_value', box_value[-4])
                if not any(elem == 0 for elem in box_value[-4]):
                    h = box_value[-5][0]
                    w = box_value[-5][1]
                    if h >= 330 or w >= 330:
                        if height_index >= 210:
                            listdata[index][-5][-1] = 0
                list_area = []
                for i in range(len(listdata)):
                    if i == index:
                        continue
                    else:
                        rect2 = listdata[i][-4]
                        rect_result = cv2.rotatedRectangleIntersection(rect1, rect2)  # picker_1 with mask_2
                        # print('rect_result', rect_result)
                        rect_result_mask = cv2.rotatedRectangleIntersection(rect1_mask, rect2)  # mask_1 with mask_2
                        if rect_result[0] == 0 and rect_result_mask[0] == 0:
                            # 无相交
                            continue
                        else:
                            if index > i-1:  # when index_box compared with heiher box, go down, or continue
                                # print('rec_result', rect_result[1])
                                if rect_result[1] is not None:
                                    # print('rect_result[1]', rect_result[1])
                                    area = cv2.contourArea(rect_result[1])
                                    # if area >= 10:
                                    cv2.polylines(self.picture, [rect_result[1].astype('int')], True, (255, 12, 0),
                                                thickness=2)
                                    list_area.append(rect_result[1])
                                    # txt = 'area:' + str(int(area)) + ' index:' + str(index) + ' i:' + str(i)
                                    # logging.info(txt)
                                    print('area:', area)
                                    # print('area:', area / ((150 / 1.4) ** 2))
                                    confidence = 0.1
                                    area_confidence_picker = area / ((150 / 1.4) ** 2)
                                    # print('stock', area_confidence_picker)
                                    if area_confidence_picker > confidence:
                                        # if area > 2700:
                                        # print('stock', area_confidence_picker)
                                        logging.warning('PICKER AREA > CONFIDENCE')
                                        obstacle_flag = False
                                if rect_result_mask[1] is not None:
                                    # print('rect_result_mask[1]', rect_result_mask[1])
                                    # list_area.append(rect_result_mask[1])
                                    area = cv2.contourArea(rect_result_mask[1])
                                    # list_area.append(rect_result_mask[1])
                                    # print('area' , area)
                                    cv2.polylines(self.picture, [rect_result_mask[1].astype('int')], True,
                                                  (123, 211, 0),
                                                  thickness=2)
                                    # txt = 'maskarea:' + str(int(area)) + ' index:' + str(index) + ' i:' + str(i)
                                    # logging.info(txt)
                                    if area > 1000:
                                        obstacle_flag = False
                                # listdata[index][-3] = [list_area]
                                # print('listdata[index][-3]', listdata[index][-3])
                            else:
                                continue
                    if obstacle_flag is False:
                        break
                # print(index, obstacle_flag)
                if obstacle_flag is False:
                    listdata[index][-5][-1] = 0
                else:
                    self.draw_picker(rotated_points, rotated_catch_list, style)
            # print('listdata', listdata[index])
                # print('list_area', list_area)
            if list_area is not None and len(list_area) > 0:
                listdata[index][-3] = list_area
            # print('listdata[index][-3]', listdata[index][-3])
                
            return listdata
        else:
            return listdata
