import numpy as np
import os
import h5py
import glob
import re
import cv2 as cv
import copy
import matplotlib.pyplot as plt
import re
import colorsys
import random
from skimage.measure import find_contours
from matplotlib.patches import Polygon


data_path = r'E:\E\MRI-CT\label\doctor\CT\original'
data_path2 = r'E:\E\MRI-CT\label\doctor\CT\smm\val'
#Saved path
target_path = './data_train_cut_padding'
eps = 0.0001


def get_brain_region(image):
    # get the brain region
    indice_list = np.where(image > 0)
    # calculate the min and max of the indice,  here volume have 3 channels
    channel_0_min = min(indice_list[0])
    channel_0_max = max(indice_list[0])

    channel_1_min = min(indice_list[1])
    channel_1_max = max(indice_list[1])

    brain_volume = image[channel_0_min:channel_0_max, channel_1_min:channel_1_max]
    # broad 5 pixels
    return (channel_0_min, channel_0_max, channel_1_min, channel_1_max)


# Transfer CT slices and corresponding labels to H5 file
def brain_png_h5_slices(data_path, target_path, num):
    '''
        每num个切片数据以及其对应的label，生成一个h5文件
        num: 测试切片数量
    '''
    img_path = os.path.join(data_path, "brain")
    label_path = os.path.join(data_path, "label")
    patients_img = os.listdir(img_path)
    patients_l = os.listdir(label_path)
    assert len(patients_img) == len(patients_l)
    count = 0

    for n, patient in enumerate(patients_l):
        img_vol_path = os.path.join(img_path, patient)
        label_vol_path = os.path.join(label_path, patient)
        imgs = os.listdir(img_vol_path)

        slices_img_list = []
        slices_label_list = []
        slice_count_list = []
        if n==4:
            ad = 1
        for i in range(0, len(imgs) - 4):
            slice_path = os.path.join(img_vol_path, "{}.png".format(i))
            label_slice_path = os.path.join(label_vol_path, "{}.png".format(i))
            img1 = cv.imread(slice_path, cv.IMREAD_GRAYSCALE)
            label1 = cv.imread(label_slice_path, cv.IMREAD_GRAYSCALE)
            # 抽出正样本
            # if np.sum(label1) != 0:
            #     print("subject:{}  slice:{}".format(n,i))
            #     continue

            # 将脑部抠出
            coord = get_brain_region(img1)
            out_img = img1[coord[0]:coord[1], coord[2]:coord[3]]
            out_label = label1[coord[0]:coord[1], coord[2]:coord[3]]
            # log.write(str(coord[1]-coord[0])+"  "+str(coord[3]-coord[2])+"\n")
            # 测试将抠出的图像center-croppping and padding to 400*400
            h, w = out_img.shape
            CROP_SIZE = 400
            if w < CROP_SIZE or h < CROP_SIZE:
                # zero cropping
                pad_h = (CROP_SIZE - h) if (h < CROP_SIZE) else 0
                pad_w = (CROP_SIZE - w) if (w < CROP_SIZE) else 0
                rem_h = pad_h % 2
                rem_w = pad_w % 2
                pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)
                pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)
                # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
                npad = (pad_dim_h, pad_dim_w)
                img_pad = np.pad(out_img, npad, 'constant', constant_values=0)
                label_pad = np.pad(out_label, npad, 'constant', constant_values=0)
                h, w = img_pad.shape
            else:
                img_pad = out_img
                label_pad = out_label
            # center crop
            h_offset = (h - CROP_SIZE) // 2
            w_offset = (w - CROP_SIZE) // 2
            cropped_img = img_pad[h_offset:(h_offset + CROP_SIZE), w_offset:(w_offset + CROP_SIZE)]
            cropped_label = label_pad[h_offset:(h_offset + CROP_SIZE), w_offset:(w_offset + CROP_SIZE)]

            slices_img_list.append(cropped_img)
            slices_label_list.append(cropped_label)
            slice_count_list.append(i)
            # 测试不同切片数量
            if len(slices_label_list) == num:
                slices_img_array = np.array(slices_img_list, dtype="uint8")
                slices_label_array = np.array(slices_label_list, dtype="uint8")
                slices_img_array = slices_img_array.transpose((1,2,0))
                slices_label_array = slices_label_array.transpose((1,2,0))
                # 判断是否都为背景(训练集时t=1, 验证集t取-1)
                t = -1
                if np.sum(slices_label_array) > t:
                    # print("subject:{}  slice:{}".format(n,i))
                    # cv.imshow("image1", slices_label_array[:,:,0])
                    # cv.imshow("image2", slices_label_array[:,:,1])
                    # cv.imshow("image3", slices_label_array[:,:,2])
                    # cv.waitKey(500)
                    # save the h5 file
                    #with h5py.File(os.path.join(target_path, '%s.h5' % count), 'w') as f:
                    # 命名取中间切片
                    length = len(slice_count_list)
                    with h5py.File(os.path.join(target_path, '{}_{}.h5'.format(n, slice_count_list[length//2])), 'w') as f:
                        f['subject'] = n
                        f['slice'] = slice_count_list
                        f['data'] = slices_img_array
                        f['label'] = slices_label_array
                    count += 1
                else:
                    pass
                # 顶部元素出栈，底部从新加入一个元素，进栈
                slices_img_list.pop(0)
                slices_label_list.pop(0)
                slice_count_list.pop(0)
            else:
                pass


# Skull Stripping
def brain_region_extraction(path, save_path, log):
    files = os.listdir(path)
    for i in range(len(files)):
        filepath = os.path.join(path, "{}.png".format(i))
        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        imgrgb = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        shape = img.shape
        # 1.先找到轮廓
        _, thresh = cv.threshold(img, 240, 255, cv.THRESH_BINARY)
        contours, image = cv.findContours(thresh, 1, cv.CHAIN_APPROX_NONE)
        # 对提取到所有轮廓排序（降序）根据面积大小排序
        contours1 = copy.deepcopy(contours)

        contours.sort(key=lambda x: cv.contourArea(x), reverse=True)
        contours1.sort(key=lambda x: x.size, reverse=True)
        if len(contours1)>=2:
            # 选择第二大面积对应的轮廓
            select_index = 1
             # 计算面积
            area = cv.contourArea(contours1[select_index])
            thresh_area = 5000
            contours_selected = contours1

            # 如果第二长的轮廓有问题，则选择最长的（保险起见，比较稳健）
            flag = False
            select_index, index_change_flag = test_centroid(contours_selected, select_index, shape, flag)
            # cv.drawContours(imgrgb, contours, 0, (0, 255, 0), 3)
            # cv.drawContours(img_back, contours_selected, 0, (0,0,255),3)
            epsilon = 0.01 * cv.arcLength(contours_selected[select_index], True)
            approx = cv.approxPolyDP(contours_selected[select_index], epsilon, True)
            # cv.drawContours(img_back, approx, -1, (0, 255, 0), 8)
            # cv.polylines(img_back, [approx], True, (0, 255, 0), 2)
            # cv.drawContours(img_back, contours_selected, 2, (255, 0, 0), 3)


            # cv.drawContours(imgrgb, contours_selected,  select_index, (0, 250, 0), -1)

            # 2.寻找凸包，得到凸包的角点
            hull2 = []
            hull = cv.convexHull(contours_selected[select_index], returnPoints=True)

            cv.drawContours(imgrgb, [hull], -1, color=(0, 255, 0), thickness=cv.FILLED)
            # 3.绘制凸包
            images = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            # cv.polylines(imgrgb, [hull], True, (255, 0, 0), 2)
            # cv.circle(images, point, radius=4, color=(255, 0, 0), thickness=cv.FILLED)

            # log.write(str(area)+'\n')
            print(str(area))
            # cv.putText(imgrgb, text=str(area),org=(220,320), fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=2, color= [0,0,255], thickness=3)
            slice_save_path = os.path.join(save_path,"{}.png".format(i))
            B_channel = imgrgb[:,:,0]
            G_channel = imgrgb[:, :, 1]
            sub_BG = G_channel-B_channel
            sub_BG[sub_BG<250] = 0
            sub_BG[sub_BG>=250] = 1


            ROI = sub_BG*img
            cv.imwrite(slice_save_path, ROI)
            # cv.imshow("segment", ROI)
            # cv.waitKey(1000)
        else:
            slice_save_path = os.path.join(save_path, "{}.png".format(i))
            cv.imwrite(slice_save_path, ROI)
            # cv.imshow("segment", ROI)
            # cv.waitKey(1000)


def test_centroid(contours_selected, select_index,shape, flag):
    moment = cv.moments(contours_selected[select_index])
    point = (int(moment["m10"] / (moment["m00"] + eps)), int(moment["m01"] / (moment["m00"] + eps)))
    if abs(point[0] - shape[0] / 2) > 40 or (shape[0] / 2 - point[1]) > 50 or (point[1] - shape[0] / 2) > 80:
        edit_select_index = 0
        index_change_flag = not flag
    else:
        edit_select_index = select_index
        index_change_flag = flag
    return edit_select_index, index_change_flag


def brain_region_extraction_main(save_path):
    subjects = os.listdir(path)
    with open("aera.txt", 'a') as log:
        for subject in subjects:
            filepath = os.path.join(path, subject)
            subject_save_path = os.path.join(save_path, subject)
            if not os.path.exists(subject_save_path): os.makedirs(subject_save_path)
            brain_region_extraction(filepath, subject_save_path, log)


if __name__ == '__main__':

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    path = "./Trans_PNG"
    if not os.path.exists(path):
        os.makedirs(path)

    brain_region_extraction_main(path)
    # Generate H5 file
    brain_png_h5_slices(path, target_path, 7)





