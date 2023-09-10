"""
code : utf-8
By Liu jialong
2023-08-31
"""

import cv2
import sys
import numpy as np


# 获取角点
def GetCorners(contours, threshold):
    corners = None
    for i in contours:  # 遍历轮廓
        area = cv2.contourArea(i)  # 计算轮廓面积
        if area > threshold:  # 面积大于阈值
            # 几何逼近轮廓获取最少点数
            approx = cv2.approxPolyDP(i, 0.01 * cv2.arcLength(i, True), True)
            rect = []
            for i in range(0, len(approx)):  # 遍历逼近轮廓
                rect.append(list(approx[i][0]))  # 提取角点的坐标
            if len(rect) == 4:  # 矩形
                # 绘制角点
                corners = rect
                break  # 检测到一个面积符合要求的轮廓即可停止遍历

    # 将corner坐标进行排序：0-左上，1-右上，2-左下，3-右下
    if corners is not None:  # 检测角点数量是否能构成矩形
        # 左上角的点具有最小的x+y值，右下角的点具有最大的x+y值
        xy_sum = []
        for i in corners:
            xy_sum.append(sum(i))  # 对每个坐标进行xy求和
        corners_order = [None, None, None, None]
        corners_order[0] = corners.pop(xy_sum.index(min(xy_sum)))  # 左上角
        xy_sum.remove(min(xy_sum))  # 删除左上角
        corners_order[3] = corners.pop(xy_sum.index(max(xy_sum)))  # 右下角
        # 除了左下与右上的两个点具有x方向大小的区别和y方向大小的区别，很容易判断
        if corners[0][0] > corners[1][0]:
            corners_order[1] = corners[0]  # 右上角
            corners_order[2] = corners[1]  # 左下角
        else:
            corners_order[1] = corners[1]  # 右上角
            corners_order[2] = corners[0]  # 左下角
    return corners_order


# 绘制角点
def DrawCorners(img, corners):
    corner = None
    if corners is not None:
        corner = img
        for i in corners:
            corner = cv2.circle(corner, i, 6, (0, 0, 255), 8)
    return corner


# 透视变换
def PerspectiveTransform(img, corners):
    # 将corner坐标进行排序：0-左上，1-右上，2-左下，3-右下
    # 透视变换
    warped = None
    h = round(((corners[2][1] - corners[0][1]) + (corners[3][1] - corners[1][1])) / 2)  # 高度
    w = round(((corners[1][0] - corners[0][0]) + (corners[3][0] - corners[2][0])) / 2)  # 宽度
    pst1 = np.float32(corners)
    pst2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    mat = cv2.getPerspectiveTransform(pst1, pst2)  # 获取透视变换矩阵
    warped = cv2.warpPerspective(img, mat, (w, h))  # 进行透视变换
    return warped


if __name__ == '__main__':  # 测试1
    impath = None
    show = False  # 是否显示处理过程中的图像

    # 检查是否通过命令行参数传入图片
    # 使用命令行参数
    if len(sys.argv) >= 2:
        impath = sys.argv[1]

    # 检查是否通过命令行参数传入图片
    if impath is None:
        # 调用摄像头
        cap = cv2.VideoCapture(0)
        # 读取摄像头图像
        success, img = cap.read()
    else:
        # 读取通过命令行传入的图像
        img = cv2.imread(impath)

    # 文档扫描时，paper的面积应该足够大才行，要不然扫描出来也不清楚
    # 姑且将阈值设置为相机范围的1/9
    area_threshold = (img.shape[0] * img.shape[1]) / 9  # 面积阈值

    # 图像处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    mid_val = round((int(gray.max()) + int(gray.min())) * 0.5)  # 获取图片所有最高值与最低值的中值
    mean_val = round(gray.sum() / (gray.shape[0] * gray.shape[1]))  # 获取图片所有像素值的均值
    ret, binary = cv2.threshold(gray, mid_val, 255, cv2.THRESH_BINARY)  # 二值化
    # 查找轮廓: 只找出最外围的轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 在原图的副本上绘制轮廓形状
    outline = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)
    # 获取paper的四个角点
    corners = GetCorners(contours, area_threshold)  # 获取角点
    # 在原图的副本上绘制四个角点
    corner = DrawCorners(img.copy(), corners)
    # 透视变换
    trans = PerspectiveTransform(img.copy(), corners)

    # 显示缩略图
    if show:
        img_names = ['img', 'gray', 'binary', 'outline', 'corner', 'trans']
        imgs = [img, gray, binary, outline, corner, trans]
        for i in range(len(imgs)):
            img = cv2.resize(imgs[i], None, None, fx=0.5, fy=0.5)
            cv2.imshow(str(img_names[i]), img)
        cv2.waitKey(0)
    # 保存图像
    if trans is not None:
        cv2.imwrite('./img.jpg', trans)  # 保存原图中的paper
