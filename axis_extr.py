#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c_map
from skimage import io as skimgIO
from skimage.filters import (sobel, threshold_otsu)
from skimage.transform import (hough_line, hough_line_peaks,
                   probabilistic_hough_line, resize, rotate,
                   hough_circle, hough_circle_peaks)
from skimage.exposure import histogram
from skimage.feature import canny, corner_harris, corner_peaks, corner_subpix, hog
from skimage.color import (rgb2gray, label2rgb)
from skimage.measure import (label, regionprops)
from PIL import Image
import matplotlib.patches as mpatches


def axis_extr(img_path):
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

    edge_sobel = sobel(img_gray)   # 滤波 边缘检测算子

    # 二值化
    thresh = threshold_otsu(edge_sobel)
    edge_binary = edge_sobel > thresh/3
    PI = 3.1415926
    # 基于概率的霍夫线性变换
    l_size_rate = 0.52  #检测直线长度阈值比率：只检测(线长/图像尺寸)比例大于l_size_rate的直线
    horizontal_lines = probabilistic_hough_line(edge_binary, threshold=int(edge_binary.shape[1]*l_size_rate),
                        line_length=int(edge_binary.shape[1]*l_size_rate),line_gap=3,
                        theta=np.asarray([PI/2]))
    vertical_lines = probabilistic_hough_line(edge_binary, threshold=int(edge_binary.shape[0]*l_size_rate),
                        line_length= int(edge_binary.shape[0]*l_size_rate),line_gap=3,
                        theta=np.asarray([0.0]))
    # sort
    horizontal_lines.sort(key = lambda line: line[0][1])
    vertical_lines.sort(key = lambda line: line[0][0])
    ####### remove repetition   ######
    thresh = 7
    horizontal_lines_temp = horizontal_lines.copy()
    for index in range(len(horizontal_lines)-1):
        cur_line = horizontal_lines[index]
        if cur_line not in horizontal_lines_temp:  # 避免之前的remove(next_line)操作造成的错误（避免二次剔除）
            continue
        next_line = horizontal_lines[index+1]
        if abs(cur_line[0][1]-next_line[0][1]) < thresh:
            if abs(cur_line[0][0]-cur_line[1][0]) > abs(next_line[0][0]-next_line[1][0]):
                horizontal_lines_temp.remove(next_line)
            else:
                horizontal_lines_temp.remove(cur_line)

    horizontal_lines = horizontal_lines_temp

    vertical_lines_temp  = vertical_lines.copy()
    for index in range(len(vertical_lines)-1):
        cur_line = vertical_lines[index]
        if cur_line not in vertical_lines_temp:
            continue
        next_line = vertical_lines[index+1]
        if abs(cur_line[0][0] - next_line[0][0]) < thresh:
            if abs(cur_line[0][1]-cur_line[1][1]) > abs(next_line[0][1]-next_line[1][1]):
                vertical_lines_temp.remove(next_line)
            else:
                vertical_lines_temp.remove(cur_line)

    vertical_lines = vertical_lines_temp
    ####################################

    ##### figure plot show the process
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(20, 20))
    ax = axes.ravel()   # flatten axes object

    for a in ax:
        a.set_axis_off()

    ax[0].imshow(edge_sobel, cmap = c_map.gray)
    ax[0].set_title("sobel edge")

    for line in vertical_lines + horizontal_lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))

    ax[1].imshow(np.zeros_like(edge_binary), cmap = c_map.gray)
    ax[1].set_title('Probabilistic Hough line')


    #### line filter  ####
    # first, filter the lines near by image boundaries and the lines crossing the whole image
    high,width = edge_binary.shape
    Off_bounding = 0.04    #距离图像边缘的距离阈值
    Length_rate = 0.05    #去除横跨整幅图像的line的阈值 例如去除length>(img_high or img_width)*(1-length_rate)的line
    vertical_lines_filted1 = list(filter(lambda line: line[0][0] > Off_bounding*width
                    and line[0][0] < width*(1-Off_bounding)
                    and abs(line[0][1]-line[1][1])<=high*(1-Length_rate), vertical_lines))

    horizontal_lines_filted1 = list(filter(lambda line: line[0][1] > Off_bounding*high
                    and line[0][1] < high*(1-Off_bounding)
                    and abs(line[0][0]-line[1][0])<=width*(1-Length_rate),horizontal_lines))
    '''
    for line in horizontal_lines_filted1 + vertical_lines_filted1:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].imshow(np.zeros_like(edge_binary), cmap = c_map.gray)
    ax[2].set_title('lines filter the lines near image boundaries and the lines that cross the whole image')
    '''
    # second, filter the shorter lines ,remanding the longest horizontal_lines and vertical_lines
    vertical_lines_filted1_length = list(map(lambda line: abs(line[0][1]-line[1][1]), vertical_lines_filted1))
    horizontal_lines_filted1_length = list(map(lambda line: abs(line[0][0]-line[1][0]), horizontal_lines_filted1))

    l_rate = 0.9    # 相对最长直线的比例，凡是cur_line/longest_line小于l_rate的线将被去除
    if len(vertical_lines_filted1) >= 2:
        vertical_lines_longest = max(vertical_lines_filted1_length)
        vertical_lines_filted2 = list(filter(lambda line: abs(line[0][1]-line[1][1]) > vertical_lines_longest*l_rate,
                            vertical_lines_filted1))
    else:
        vertical_lines_filted2 = vertical_lines_filted1

    if len(horizontal_lines_filted1) >= 2:
        horizontal_lines_longest = max(horizontal_lines_filted1_length)
        horizontal_lines_filted2 = list(filter(lambda line: abs(line[0][0]-line[1][0]) > horizontal_lines_longest*l_rate,
                            horizontal_lines_filted1))
    else:
        horizontal_lines_filted2 = horizontal_lines_filted1


    ## filter the long line (maybe the edge of long bar)
    if horizontal_lines_filted2:
        # the longest horizontal line
        base_line = max(horizontal_lines_filted2, key = lambda line: abs(line[0][0]-line[1][0]))
        base_line_l = min(base_line[0][0], base_line[1][0])
        base_line_r = max(base_line[0][0], base_line[1][0])
        base_line_len = base_line_r - base_line_l
        '''
        vertical_lines_filted3 = list(filter(lambda v_line: base_line_l<v_line[0][0]<base_line_l+base_line_len*1/8
                                            or base_line_r-base_line_len*1/8 < v_line[0][0] < base_line_r,
                                            vertical_lines_filted2))
        '''
        vertical_lines_filted2 = list(filter(lambda v_line: v_line[0][0]<base_line_l+base_line_len*1/8
                                            or base_line_r-base_line_len*1/8 < v_line[0][0],
                                            vertical_lines_filted2))
        
    if vertical_lines_filted2:
        base_line = max(vertical_lines_filted2, key = lambda line: abs(line[0][1]-line[1][1]))
        base_line_up = min(base_line[0][1], base_line[1][1])
        base_line_dw = max(base_line[0][1], base_line[1][1])
        base_line_len = base_line_dw - base_line_up
        '''
        horizontal_lines_filted3 = list(filter(lambda h_line: base_line_up<h_line[0][1]<base_line_up+base_line_len*1/8
                                            or base_line_dw - base_line_len*1/8 < h_line[0][1] < base_line_dw,
                                            horizontal_lines_filted2))

        '''
        horizontal_lines_filted2 = list(filter(lambda h_line: h_line[0][1]<base_line_up+base_line_len*1/8
                                            or base_line_dw - base_line_len*1/8 < h_line[0][1],
                                            horizontal_lines_filted2))



    for line in horizontal_lines_filted2 + vertical_lines_filted2:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].imshow(np.zeros_like(edge_binary), cmap = c_map.gray)
    ax[2].set_title('lines after filter')

    # candidate

    def standardization(line):
        p0, p1 = line
        min_x = min([p0[0], p1[0]])
        max_x = max([p0[0], p1[0]])
        min_y = min([p0[1], p1[1]])
        max_y = max([p0[1], p1[1]])
        return ((min_x,min_y),(max_x, max_y))

    if len(vertical_lines_filted2)>=2:
        v_candidate = [vertical_lines_filted2[0], vertical_lines_filted2[-1]]
    else:
        v_candidate = vertical_lines_filted2
    if len(horizontal_lines_filted2)>=2:
        h_candidate = [horizontal_lines_filted2[0], horizontal_lines_filted2[-1]]
    else:
        h_candidate = horizontal_lines_filted2

    y_axis = []
    if v_candidate:
        y_axis.append(v_candidate[0])
    x_axis = []
    if h_candidate:
        x_axis.append(h_candidate[-1])

    axis = [standardization(line) for line in x_axis + y_axis]
    axis_candidate = [standardization(line) for line in v_candidate + h_candidate]

    print('final candidate ')
    print(len(v_candidate),'vertical_lines')
    print(v_candidate)
    print(len(h_candidate), 'horizontal_lines')
    print(h_candidate)
    print(axis_candidate)

    for line in axis_candidate:
        p0, p1 = line
        ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[3].imshow(np.zeros_like(edge_binary), cmap = c_map.gray)
    ax[3].set_title('axies candidate')

    fig.tight_layout()
    result_root = os.path.join(os.path.dirname(img_path) , "axis_result_modify2")
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    plt.savefig( os.path.join(result_root, os.path.basename(img_path) + '_process.jpg'))
    plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax = axes.ravel()   # flatten axes object

    for a in ax:
        a.set_axis_off()

    ax[0].imshow(img_ori)
    ax[0].set_title('Input image')

    ax[1].imshow(img_ori)
    ax[1].set_title('axis candidate')

    for axis_l in axis:
        x, y = axis_l[0]
        dx,dy = axis_l[1][0] - axis_l[0][0], axis_l[1][1] - axis_l[0][1]
        axis_patch = mpatches.Arrow(x, y, dx, dy, color='r')
        ax[1].add_patch(axis_patch)

    plt.savefig( os.path.join( result_root, os.path.basename(img_path) + '_result.jpg'))
    plt.close(fig)

    return axis

if __name__ == "__main__":
    #img_path = '/home/hliu/euro_vis/chart_component_extract/data_test/bar_img_failed/real_bar_34.png'
    img_root = '/home/hliu/euro_vis/chart_component_extract/data_test/line_img_failed'
    #img_root = '/media/hliu/lhuan/euro_vis/evaluation_data/evalset_fqa/pie/bitmap'
    #result_root = os.path.join(os.path.dirname(img_root), os.path.basename(img_root)+'_axis_result')
    
    #if not os.path.exists(result_root):
        #os.mkdir(result_root)

    file_list = os.listdir(img_root)
    for cur_name in file_list:
        
        img_path = os.path.join(img_root, cur_name)
        assert os.path.exists(img_path),  "the image file not exists"
        print(img_path)
        
        axis = axis_extr(img_path)
        print(axis)

