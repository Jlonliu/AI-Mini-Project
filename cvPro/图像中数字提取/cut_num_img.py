import cv2 as cv
import numpy as np
import sys
import pickle

# import argparse
# import matplotlib.pyplot as plt


# 二值化图像反转（黑变白，白变黑）
def BinaryInverse(img):
    img[img == 0] = 100  # 将原图中黑色部分变成灰色
    img[img == 255] = 0  # 将原图中白色部分全部变成黑色
    img[img == 100] = 255  # 将变换后的图片中的灰色变成白色
    return img


if __name__ == '__main__':  # 测试1
    # 使用命令行参数
    if len(sys.argv) < 2:
        print("Usage: python ocr.py image.png")
        sys.exit(1)
    impath = sys.argv[1]

    # *******************************模板图像处理************************************
    # temp_path = "./template/number_temp.png"
    temp = cv.imread(impath, cv.IMREAD_COLOR)
    temp_gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)  # 获取灰度图
    temp_h, temp_w, channals = temp.shape  # 获取原图尺寸

    # 图像轮廓
    mid_val = int((int(temp_gray.max()) + int(temp_gray.min())) * 0.5)  # 获取图片所有最高值与最低值的中值
    # mean_val = int(temp_gray.sum() / (temp_gray.shape[0] * temp_gray.shape[1]))  # 获取图片所有像素值的均值
    ret, binary = cv.threshold(temp_gray, mid_val, 255, cv.THRESH_BINARY)  # 二值化
    binary = BinaryInverse(binary)  # 反转二值化图像，变为黑低白字
    # temp_zero = np.zeros((temp_h, temp_w, channals), np.uint8)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # res = cv.drawContours(temp_zero, contours, -1, (0, 0, 255), 1)
    rect_list = []
    for i in range(0, len(contours)):
        # print(rect)
        rect = cv.boundingRect(contours[i])
        rect_list.append(rect)
        # if (rect[0]+rect[1])>contours_list
        # cv.rectangle(temp, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)
    temp_sort = sorted(rect_list)  # 将位置信息从小到大排序，先按x排序，再在x排序的基础上对y排序
    # num_sort = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    # *******************************************************************************

    # # *******************************对象图像处理************************************
    # img = cv.imread(impath, cv.IMREAD_COLOR)
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 获取灰度图
    # img_h, img_w, channals = img.shape  # 获取原图尺寸

    # # 图像轮廓
    # mid_val = int((int(temp_gray.max()) + int(temp_gray.min())) * 0.5)  # 获取图片所有最高值与最低值的中值
    # mean_val = int(img_gray.sum() / (img_gray.shape[0] * img_gray.shape[1]))  # 获取图片所有像素值的均值
    # ret, binary = cv.threshold(img_gray, mid_val, 255, cv.THRESH_BINARY)  # 二值化
    # binary = BinaryInverse(binary)  # 反转二值化图像，变为黑低白字
    # # temp_zero = np.zeros((temp_h, temp_w, channals), np.uint8)
    # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # # res = cv.drawContours(temp_zero, contours, -1, (0, 0, 255), 1)
    # rect_list = []
    # for i in range(0, len(contours)):
    #     rect = cv.boundingRect(contours[i])
    #     rect_list.append(rect)
    #     # if (rect[0]+rect[1])>contours_list
    #     cv.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)
    # split_sort = sorted(rect_list)  # 将位置信息从小到大排序，先按x排序，再在x排序的基础上对y排序

    # # 匹配
    # for i in split_sort:  # 遍历检测图像
    #     img_temp = img_gray[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])]  # 获取分割后的数字图像
    #     score = []
    #     for t in temp_sort:  # 遍历模板图像
    #         temp_temp = temp_gray[t[1]:(t[1] + t[3]), t[0]:(t[0] + t[2])]
    #         img_temp_resize = cv.resize(img_temp, (t[2], t[3]), interpolation=cv.INTER_LINEAR)  # 将检测图像大小设置为模板图像一样大小
    #         res = cv.matchTemplate(img_temp_resize, temp_temp, cv.TM_SQDIFF)
    #         min_val, max_val, min_loc, mak_loc = cv.minMaxLoc(res)
    #         score.append(min_val)
    #     index = score.index(min(score))     # 获取最小值的下标
    #     # 将文字输出为cv2的字体，并显示在图片上
    #     cv.putText(img, str(num_sort[index]), (i[0], i[1]), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    #     # print(num_sort[index])  # 输出对应的数字
    #     print(index)

    rst_list = []
    img_show = np.zeros((30, 30 * len(temp_sort)), np.uint8)
    for i in range(0, len(temp_sort)):  # 遍历查找到的数字图像坐标
        rect = temp_sort[i]
        # 根据坐标将数字图像截取出来
        cut_img = binary[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        # 获取数字图像的长宽
        h, w = cut_img.shape[0], cut_img.shape[1]
        max_val = max(h, w)  # 获取数字图像长宽的最大值
        # 将数字图像变为正方形图像，多出来的部分用0填充
        mnist_img = cv.copyMakeBorder(cut_img,
                                      int((max_val - h) / 2),
                                      int((max_val - h) / 2),
                                      int((max_val - w) / 2),
                                      int((max_val - w) / 2),
                                      borderType=cv.BORDER_CONSTANT,
                                      value=0)
        mnist_img = cv.resize(mnist_img, (20, 20))  # 将图像变为20*20的小图像
        # 在四周填充上0，以便让数字位于图像中心，最终图像大小为28×28，符合mnist数据集中图像的大小要求
        mnist_img = cv.copyMakeBorder(mnist_img, 4, 4, 4, 4, borderType=cv.BORDER_CONSTANT, value=0)
        # cv.imshow(str(i),img5)
        rst_list.append(mnist_img)  # 将数字图像存储到列表中
        show_img = cv.copyMakeBorder(mnist_img, 1, 1, 1, 1, borderType=cv.BORDER_CONSTANT, value=255)
        img_show[0:30, 30 * i:30 * (i + 1)] = show_img
    cv.imshow("Cuted", img_show)

    # 生成pickle文件存储数字图像，以便其他python程序调用
    with open("./my_mnist_test_img.pkl", 'wb') as f:
        pickle.dump(rst_list, f, -1)
    print("Done!")
    # 字体颜色的最大值240,最小值90
    # cv.imshow("binary", binary)
    # cv.imshow("Source", img)
    # cv.imshow("Result", temp)
    cv.waitKey(0)
