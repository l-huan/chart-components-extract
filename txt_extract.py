#!/usr/bin/env python
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


img_path = './data_test/bar_img_failed/real_bar_19.png'
img_ori = skimgIO.imread(img_path)   #<class 'imageio.core.util.Array'>
#plt.imshow(img_ori)
type(img_ori.dtype)
plt.imshow(img_ori)
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

    # 二值化
    thresh = threshold_otsu(img_gray)
    img_binary = img_gray < thresh

    label_img, label_num = label(img_binary, background=None, return_num=True, connectivity=2)

    chara_bbox_cent = []
    chara_bbox = []
    label_rm = []
    img_height, img_width = img_binary.shape
    bbox_thresh = img_width*img_height//32000
   
    for region in regionprops(label_img):                 
        # take regions with large enough areas and remove region 离心率大于0.993的区域（线状区域）
        if (img_height*img_width) // 40 >=region.bbox_area >= bbox_thresh \
                and region.eccentricity < 0.993:
            # remove area/凸包面积大于0.97的区域（整体的块状区域，无孔洞）
            if (region.area > 40 and region.solidity <0.9) or region.area <=40:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=1)

                chara_bbox_cent.append(region.centroid)
                chara_bbox.append(region.bbox)
                #chara_bbox_area.append(region.bbox_area)
                #chara_area.append(region.area)
            else:
                label_rm.append(region.label)
        else:
            label_rm.append(region.label)


    # 将去除干扰元素后的字母元素提取出来
    label_rm.append(0)
    img_crop = np.isin(label_img, label_rm, invert=True)

    # 腐蚀操作
    
    kernel_size = 6
    if img_height*img_width < 400000:
        kernel_size = 5
    if img_height*img_width < 200000:
        kernel_size = 4
    if img_height*img_width < 100000:
        kernel_size = 3
    if img_height*img_width < 70000:
        kernel_size = 2
    
    print("*****************", kernel_size, "*******************")
    kernel = morphology.disk(kernel_size)
    img_dialtion = morphology.dilation(img_crop, kernel)

    # 腐蚀后再次连通域检测
    label_txt, label_num = label(img_dialtion, background=None, return_num=True, connectivity=2)

    txt_bbox_cent = []
    txt_bbox = []

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img_ori)
    for region in regionprops(label_txt):  
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=1)
        
        txt_bbox_cent.append(region.centroid)
        txt_bbox.append(region.bbox)
        ax.add_patch(rect)

    plt.savefig(os.path.join(result_img_dump, os.path.basename(img_path) + "_txt_bbox.png"))
    plt.close(fig)
    return txt_bbox    

if __name__ == "__main__":
    #img_root = '/home/lhuan/chart-components-extract/data_test/line_img_failed'
    #img_root = '/home/lhuan/chart-components-extract/data_test/bar_img_failed'
    #img_root = '/media/lhuan/lhuan/euro_vis/evaluation_data/evalset_fqa/vbar/bitmap'
    img_root = '/media/lhuan/lhuan/euro_vis/evaluation_data/web_collected_test/real_pie/image'

    result_root = img_root+'_txtbox_result'  
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    file_list = os.listdir(img_root)
    for cur_name in file_list:      
        img_path = os.path.join(img_root, cur_name)
        if os.path.isfile(img_path) :
            print(img_path)           
            txt_bbox = txt_bbox_detection(img_path, result_root)
            print(txt_bbox)
    


'''

txt_ocr = []
for i, bbox in enumerate(txt_bbox):
    pre_img_array = img_ori[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    temp_img = Image.fromarray(pre_img_array)
    temp_img.save("txt_bbox_img/"+str(i)+".png")



# ocr文本识别
import pyocr
import pyocr.builders as ocrtools
ocr_tool = pyocr.get_available_tools()[0]
assert tool is not None, "no ocr_tool acailable!!!"

txt_ocr = []
for bbox in txt_bbox:
    # 将竖向文本旋转90度  TODO
    
    pre_img_array = img_ori[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    txt = ocr_tool.image_to_string(
                Image.fromarray(pre_img_array),
                lang="eng",
                builder=ocrtools.DigitBuilder())
    
    txt_ocr.append(txt)


'''

