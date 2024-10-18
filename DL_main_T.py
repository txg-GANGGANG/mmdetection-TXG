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

sum=0
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
if not os.path.exists(img_path + '\\{}'.format(time.strftime("%Y-%m-%d"))):
    os.mkdir(img_path + '\\{}'.format(time.strftime("%Y-%m-%d")))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=img_path + '\\{}\\{}.txt'.format(time.strftime("%Y-%m-%d"), time.strftime("%Y-%m-%d")),
                    filemode='a')

from D3_T import Height
from Picture_T import Picture

config_file = r'D:\testAI\mmdetection-master\mask_rcnn_r101_fpn_1x_coco.py'
# checkpoint_file = r'D:\testAI\kfw\biandang\epoch_500.pth'
# config_file = r'E:\testAI\mmdetection\configs\mask_rcnn\mask_rcnn_r101_fpn_1x_coco.py'
# checkpoint_file = r'E:\testAI\mmdetection\tools\work_dirs\mask_rcnn_r101_fpn_1x_coco\epoch_900.pth'
# config_file = r'E:\testAI\mmdetection_KFW\mask_rcnn_r101_fpn_1x_coco.py'

global time_, path, count1,inf_time, lentho, widtho, heighto, count_list_data
time_ = ''
count1 = 0
lentho = 0
widtho = 0
heighto = 0
count_list_data = 0
list_center_put=[]
center_put = []
list_data2 = []
path = img_path + '\\{}'.format(time.strftime("%Y-%m-%d")) + '\\' + str(count1)
model = init_detector(config_file, checkpoint_file, device='cpu')
halcon_array = np.array([[0.259637,-0.965701,-0.003083,-475.648],
                        [-0.963062,-0.258689,-0.0747824,-31.9998],
                        [0.0714199,0.0223854,-0.997195,-143.15],
                        [0.0,0.0,0.0,1.0]])
# halcon_array2 = np.array([[-0.244611,0.944747,0.218215,201.162],
#                         [0.96838,0.226649,0.104258,97.5291],
#                         [0.0490393,0.236818,-0.970316,-83.8744],
#                         [0.0,0.0,0.0,1.0]])


def inf(image, image_h):
    logging.info('inf begin')
    logging.info('\n')
    global path, count1, area, lentho, widtho, heighto,count_put
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in range(10000):
        paths = os.listdir(path)
        if len(paths) >= 500:
            count1 += 1
            path = path + str(count1)
            if not os.path.isdir(path):
                os.mkdir(path)
        else:
            break
    style1 = 0
    # [[X,Y,Z,Rz,style,H,W],[num]]
    # stock_retrigger = [[0, 0, 0, 0, 0, 0, 0], [0]]
    # stock_supertall = [[7, 7, 7, 7, 7], [1]]
    stock_pick = []
    logging.info('new_picture_begin')
    # logging.info('signal:' + str(signal))
    ct = time.time()
    data_secs = (ct - int(ct)) * 1000
    global time_
    time_ = time.strftime("%Y-%m-%d %H-%M-%S")
    time_ = "%s.%03d" % (time_, data_secs)
    h_indicate = []  # output: x, y, z, or rz
    # img = mmcv.imread(image)
    # img = image
    # print(image.shape)``
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    origin_shape = (image.shape[0], image.shape[1])  
    logging.info('shape:' + str(image.shape))
    if image.shape[2] == 1:
        logging.info('gray2rgb')
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2RGB)

    logging.info('shape:' + str(image.shape))
    # img = img.astype('uint8')

    Img = Picture(image, time_, path)
    H_image = Height(image_h)
    # if issave_img is True:
    #     Img.picture_save(add_name='-ori', picture=Img.picture)
    
    inf_time = 0
    stock_retrigger = [[0, 0, 0, 0, 0, 0, 0], [0]]
    logging.info('ready to detect....')
    result_detected = inference_detector(model, image)
    logging.info('detecting....')
    print('result_detected',result_detected)
    if isinstance(result_detected, tuple):
        bbox_result, segm_result = result_detected
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result_detected, None
    bboxes = np.vstack(bbox_result)
    print(bboxes)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    # print(bboxes)
    labels = np.concatenate(labels)
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        # print('111111111111111111')
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            # logging.warning('segms is tensor')
        else:
            segms = np.stack(segms, axis=0)
           
            logging.info('get rude segms, starting processing...')
            mask = np.array(segms)
            img_zeros = np.zeros(origin_shape)
            over_the_num = []
            list_data = []
            # print('labels:', labels)
            # indices = cv2.dnn.NMSBoxes(bboxes[:, :-1].tolist(), bboxes[:, -1].tolist(), score_threshold=0.6,nms_threshold=0.6, top_k=5)
            # if len(indices) == 0:
            #     logging.info('indices number = 0')
            #     return stock_retrigger
            # indices_array = indices.flatten(0)
            # print('mask',indices)
            # calss,num,height
            # num_catch,catch_height = get_num_height(signal=signal)#num_catch需要抓取的数量，catch_height需要抓取的物体高度
            # if signal != 111 and num_catch != 0 and catch_height != 0:
        
            for bb_num, bb_value in enumerate(bboxes):
                # print('bb_value--',bb_value)
                # if bb_num in indices_array:
                # Img.draw_bbox(bb_value, round(bb_value[4], 3))
                # if bb_value[4] > 0.75 and labels[bb_num] == 9:
                if bb_value[4] > 0.75:

                    tem = []
                    # print('bb_value',bb_value)
                    over_the_num.append(bb_num)
                    tem = bb_value.tolist()
                    tem.append(bb_num)
                    tem.append(labels[bb_num])
                    tem.append(-5)  # set_pickerpoints_position
                    # print('labels[bb_num]',labels[bb_num])
                    list_data.append(tem)
                else:
                    # logging.warning('Box is null')
                    continue
        
            # No box to be detected
            
            if len(list_data) == 0:
                logging.info('no can catched')
                return stock_retrigger
            logging.info('list_data_len:' + str(len(list_data)))
            colors = [num * 10 + 10 for num in range(len(over_the_num))]
            """yxz_list include every bbox point data combined with 3D camera data"""
            # cv2.rectangle(Img.picture, (150, 100), (500, 360), (0, 255, 0), 1, 0)
            # print('list_data', list_data)
            for index, bbox1 in enumerate(list_data):
                # center = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
                img_zeros += mask[bbox1[5]] * colors[index]
                d_s_result = Img.draw_segs(mask[bbox1[5]])
                if d_s_result is not None:
                    angle, box, rect, area1 = d_s_result
                else:
                    return stock_retrigger
                c_x = rect[0][0]
                c_y = rect[0][1]#mask区域的旋转中心点
                center = [c_x, c_y]
                hws_list, valueZ, valueXYZ = H_image.get_widhei_style(mask=mask[bbox1[5]], box=box, center=center)
                # r_style = robot_style(signal)# 抓手类型
                # print('c_signal'+str(signal))
                # c_style = catch_style(signal)
                # catch_l =[hws_list[-1],c_style,catch_height]
                catch_l =[hws_list[-1],0,0]

                # print('valueXYZ',valueXYZ)
                list_data[index].append(angle)
                list_data[index].append(box)
                list_data[index].append(catch_l)
                list_data[index].append(rect)
                list_data[index].append(center)
                list_data[index].append(valueXYZ)
                list_data[index].append(valueZ)
                print('list_data',list_data)
                logging.warning('list_data is ok')
            if issave_img is True:
                Img.picture_save(add_name='-rudemask-', picture=img_zeros)  # save mask image
            """important estimate"""
            # estimate hight whether be able to capture, too high or too small is not in range of capture
            """important estimate"""
            """to estimate height whether be able to pick or go down / up"""
            if len(list_data) == 0:
                logging.info('len_list_data is 000')
                return stock_retrigger
            # print('list_data--', list_data)
            # logging
            # logging.info('list_data----1' + str(list_data))
            list_data.sort(key=operator.itemgetter(-1), reverse=True)
            l = []
            st = []
            count = 0
            mask_area2_list = []
            for value in list_data:
                # boxes = value[9]
                cv2.circle(Img.picture, (int(list(value[-4][0])[0]), int(list(value[-4][0])[1])),
                            5, (0,255,0), -1)
                count += 1
                camera_point = np.append(np.array([value[-2][1], value[-2][0], value[-2][2]]), 1.0)
                logging.info((str(camera_point)))
                a = np.transpose(camera_point)
                c = halcon_array.dot(a)[0:3]
                l.append(int(c[0]))#X
                l.append(int(c[1]))#Y
                l.append(int(c[-1]))#Z
                l.append(int(value[-7]))#角度
                l.append(int(value[-5][0]))#策略
                l.append(int(value[-5][1]))#物体放置类型
                l.append(int(value[-5][2]))#物体放置高度
                if count >= 1:
                    break
                # else:
                #     continue
            """after got box_order_point, according to the corner H_image data to get pick point"""
            # st = 'num:'+str(count)+';'+'501'+'('+l[0]+')E'
            st.append(l)
            st.append([count])#数量
            logging.info('st  : ' + str(st))
            # [[x,y,z,angle,style,h,w],[num]]
            if issave_img is True:
                Img.picture_save(add_name='-pickbox-', picture=Img.picture)
            if issave_img is True:
                H_image.himg_save(path, time_)
            logging.warning('11-11-11-11')
            # print('st', st)
            return st
    else:
        logging.info('without any detection!!!!')
        return stock_retrigger

def get_num_height(signal):  
    catch_list = [[0,2,40],[1,2,40],[2,2,40],[3,2,40],[4,2,40],[5,1,70],[6,2,140],[7,2,30],[8,2,40]]#物料信息：【类名，数量，高度】
    for item in catch_list:  
        if item[0] == signal:  
            return item[1], item[2]  # 返回num和height  
    return 0, 0  # 如果没有找到对应的signal，则返回0

def catch_style(signal):
    # print('signal begian'+str(signal))
    # print('type of signal'+str(type(signal)))
    if signal == 0 or signal == 1:
        catch_style = 3
    elif signal == 2:
        catch_style = 1
    elif signal == 3:
        catch_style = 2
    elif signal == 4 or signal == 5:
        catch_style = 4
    elif signal == 6 or signal== 7 or signal==8:
        catch_style = 5
    logging.info('catch_style is '+str(catch_style))
    # print('catch_style'+str(catch_style))
    return catch_style
#1
# def inf(image, image_h, signal):
#     logging.info('hanhan')
#     logging.info(str(image[2][2]))
#     logging.info(str(image_h[2][2]))
#     # img = cv2.imread(image)
#     # img_h = tiff.imread(image_h)
#     # Img = Picture(image, time_, path)
#     # H_image = Height(image_h)
#     res = [[-437, -182, -1143, 0, 1, 252, 176], [1]]
#     return res

def test():
    import glob
    import time
    dir_path = r'D:\testAI\mmdetection-master\Images\test'
    pathlist = glob.glob(dir_path + '/*.jpg')
    print(pathlist)
    for img_path in pathlist:
        img = cv2.imread(img_path)
        # print(len(img.shape),11111111111111111111111111)
        # if len(img.shape) != 3:  
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # print(img_path)
        img_h_path = img_path.split('.')[0] + '.tiff'
        print(img_h_path)
        image_h = tiff.imread(img_h_path)
        for i in range(1):
            start = time.time()
            # result = inf(img, image_h, signal=1,b_h=0,b_w=0,b_z=0)
            result = inf(img, image_h)
            # time.sleep(1)
            # result = inf(img, image_h, signal=1)
            end = time.time()
            time.sleep(1)
            print(end - start)
            print('res:', result)

test()
