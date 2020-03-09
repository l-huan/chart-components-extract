# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
import matplotlib.patches as mpatches
from PIL import Image
from skimage import io as skimgIO
from skimage.filters import (sobel, threshold_otsu)
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line, resize, rotate,
                               hough_circle, hough_circle_peaks)
from skimage.exposure import histogram
from skimage.feature import canny, corner_harris, corner_peaks, corner_subpix, hog
from skimage.color import (rgb2gray, label2rgb)
from skimage.measure import (label, regionprops)
from skimage import morphology
import cv2 as cv



def txt_bbox_detection(img_path, result_img_dump):
    try:
        '''
        IMG = Image.open(img_path)
        img_ori = np.asarray(IMG, dtype='uint8')  # 8 bit RGB
        '''
        img_ori = skimgIO.imread(img_path)
    except ValueError:
        fp = open('log_imgIO.txt', 'a')
        fp.writelines( img_path )
        fp.close()
        return([])
    #img_ori = skimgIO.imread(img_path)   #<class 'imageio.core.util.Array'>
    img_gray = rgb2gray(img_ori)   #<class 'numpy.ndarray'>

    ##############   灰度图直接进行阈值分割，二值化    ################
    # 二值化
    thresh = threshold_otsu(img_gray)
    img_binary = img_gray < thresh
    img_height, img_width = img_binary.shape
    
    # closeing
    kernel_size = 4
   
    if img_width*img_height < 400000:
        kernel_size = 3
    if img_width*img_height < 200000:
        kernel_size = 2
    if img_width*img_height < 100000:
        kernel_size = 1
    
    img_close = morphology.closing(img_binary,selem=np.ones((kernel_size,kernel_size), dtype=np.uint8))

    label_img, label_num = label(img_close, background=None, return_num=True, connectivity=2)

    image_label_overlay = label2rgb(label_img, image=img_binary)
    
    label_rm = []
    bbox_thresh = img_width*img_height//32000

    for region in regionprops(label_img):                 
        # take regions with large enough areas and remove region 离心率大于0.993的区域（线状区域）
        if (img_height*img_width) // 40 >=region.bbox_area >= bbox_thresh \
                and region.eccentricity < 0.994:
            # remove area/凸包面积大于0.97的区域（整体的块状区域，无孔洞）
            if (region.area > 40 and region.solidity <0.9) or region.area <=40:
                # draw rectangle around segmented coins
                '''
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=1)
                                
                '''

            else:
                label_rm.append(region.label)
                
            if region.area < 10:
                label_rm.append(region.label)

        else:
            label_rm.append(region.label)


    # 将去除干扰元素后的字母元素提取出来
    label_rm.append(0)
    img_crop = np.isin(label_img, label_rm, invert=True)

    # 腐蚀操作    
    kernel_size = 5
    if img_height*img_width < 400000:
        kernel_size = 4
    if img_height*img_width < 200000:
        kernel_size = 3
    if img_height*img_width < 100000:
        kernel_size = 2
    if img_height*img_width < 70000:
        kernel_size = 1
    
    print("*****************", kernel_size, "*******************")
    kernel = morphology.disk(kernel_size)
    img_dialtion1 = morphology.dilation(img_crop, kernel)
    ##################  灰度图文本检测 end    ####################


    #################  利用边缘进行文本检测   ##################
    # 二值化
    img_sobel = sobel(img_gray)
    thresh = threshold_otsu(img_sobel)
    img_binary = img_sobel > thresh*0.94
    # closeing
    kernel_size = 4
    
    if img_height*img_width < 400000:
        kernel_size = 3
    if img_height*img_width < 200000:
        kernel_size = 2
    if img_height*img_width < 100000:
        kernel_size = 1
    
    img_close = morphology.closing(img_binary,selem=np.ones((kernel_size,kernel_size), dtype=np.uint8))
    label_img,label_num = label(img_close, background=None, return_num=True, connectivity=2) 
    
    image_label_overlay = label2rgb(label_img, image=img_binary)

    label_rm = []
    bbox_thresh = img_width*img_height/12000
    # 根据region属性筛选连通域
    for region in regionprops(label_img):
            # take regions with large enough areas
        
        if (img_height*img_width) // 40 >=region.bbox_area >= bbox_thresh \
                    and region.eccentricity < 0.994:
                
            #if (region.area > 40 and region.solidity <0.9) or region.area <=40:
            
            #去除方框
            if region.solidity < 0.4 and region.convex_area/region.bbox_area >= 0.85:
                label_rm.append(region.label)
                
            #去除小线段
            if region.area == region.convex_area:
                label_rm.append(region.label)
            
            if region.area >=10:
                # draw rectangle around segmented coins
                '''
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=1)
                                
                '''

            else:
                label_rm.append(region.label)
        else:
            label_rm.append(region.label)

    # 将去除干扰元素后的字母元素提取出来
    label_rm.append(0)
    img_crop = np.isin(label_img, label_rm, invert=True)
    
    # 腐蚀操作
    kernel_size = 4
    if img_width*img_height < 400000:
        kernel_size = 3
    if img_width*img_height < 200000:
        kernel_size = 2
    if img_width*img_height < 100000:
        kernel_size = 1
    print("*****************", kernel_size, "*******************")
    kernel = morphology.disk(kernel_size)
    img_dialtion2 = morphology.dilation(img_crop, kernel)

    ####################  边缘信息检测文本 end   #################

    
    ######################## 结合两者检测结果  #######
    img_dialtion = np.logical_or(img_dialtion1, img_dialtion2)

    # 再次连通域检测
    label_txt, label_num = label(img_dialtion, background=None, return_num=True, connectivity=2)
    
    #文本框识别结果img
    img_show = cv.cvtColor(img_ori[:,:,:3], cv.COLOR_RGB2BGR)

    txt_bbox_cent = []
    txt_bbox = []
    txt_label = []
    txt_convex_image = []
    txt_minRect_info = []
    #txt_img_forOCR = []
    
    for region in regionprops(label_txt):  
        # save some information
        txt_bbox_cent.append(region.centroid)
        txt_bbox.append(region.bbox)
        txt_label.append(region.label)
        txt_convex_image.append(region.convex_image)
        
        # 根据最小凸包的面积和bbox面积比确定文本是否倾斜
        if region.convex_area/region.bbox_area < 0.6:
            #利用opencv 最小外接矩形来检测文本框并获取矫正角度    
            rect = cv.minAreaRect(np.argwhere(label_txt.T == region.label)) 
            # note：opencv的x，y轴分别对应图像的column和row。 
            #注意这里label_txt的转置，目的是使得图像中像素的行列值与opencv的xy坐标值对应， 否则会出现怪异的结果
            #最终返回的描述最小外接矩形的（中心(x,y), (宽,高), 旋转角度），其中xy是opencv下的xy轴坐标，其中x对应图像的column，y对应row
            
            # draw rectangle in red color
            box = cv.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标
            box = np.int64(box)
            cv.drawContours(img_show, [box], 0, (0, 0, 255), 2)
            #cv.polylines(img, pts=[box], isClosed=True, color=(0,0,255), thickness=1)
            
            #### 需要返回的region局部最小外接矩形的特征描述
            local_minRect = cv.minAreaRect(np.argwhere(region.convex_image.T))
            txt_minRect_info.append(local_minRect)
            ####
        else:
            minr, minc, maxr, maxc = region.bbox
            # 判断竖直文本
            if (maxr-minr)/(maxc-minc)>2:
                # minRect_temp ：（中心(x,y), (宽,高), 旋转角度）
                minRect_temp = ((region.local_centroid[1],region.local_centroid[0]),(maxr-minr, maxc-minc), -90)
                # draw rectangle on img_show
                cv.rectangle(img_show, (minc,minr), (maxc, maxr), (0,255,0), 2)
            else:
                minRect_temp = ((region.local_centroid[1],region.local_centroid[0]),(maxc-minc, maxr-minr), 0)
                # draw 
                cv.rectangle(img_show, (minc,minr), (maxc, maxr), (0,0,255), 2)
            # 存储region局部最小外接矩形的特征描述
            txt_minRect_info.append((minRect_temp))
         
    if result_img_dump:
        fig, ax = plt.subplots(figsize=(10,8))
        ax.imshow(cv.cvtColor(img_show, cv.COLOR_BGR2RGB))
        plt.savefig(os.path.join(result_img_dump, os.path.basename(img_path) + "_txt_bbox.png"))
        plt.close(fig)

    return cv.cvtColor(img_show, cv.COLOR_BGR2RGB), txt_bbox, txt_minRect_info   
    

if __name__ == "__main__":
    #img_root = '/home/lhuan/chart-components-extract/data_test/line_img_failed'
    #img_root = '/home/lhuan/chart-components-extract/data_test/bar_img_failed'
    #img_root = '/media/lhuan/lhuan/euro_vis/evaluation_data/evalset_fqa/vbar/bitmap'
    #img_root = '/media/hliu/lhuan/euro_vis/evaluation_data/web_collected_test/real_pie/image'
    img_root = "data_test/web_collected_test/all"
    #img_root = "data_test/evalset_fqa/all"

    result_root = img_root+'_txtbox_combine3'  
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    file_list = os.listdir(img_root)
    for cur_name in file_list:      
        img_path = os.path.join(img_root, cur_name)
        if os.path.isfile(img_path) :
            print(img_path)           
            txt_bbox_detection(img_path, result_root)
            
    
