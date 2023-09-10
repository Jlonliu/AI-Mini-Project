import cv2 as cv
import numpy as np
import argparse as ap

# import matplotlib.pyplot as plt


# 检测特征点并描述
def DetetAndDescribe(img):
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 获取灰度图
    sift = cv.xfeatures2d.SIFT_create()  # 建立sift生成器
    # kps是所有关键点的坐标，features是关键点对应的128维特征向量
    kps, features = sift.detectAndCompute(img, None)  # 检测关键点并计算描述特征
    # kps是opencv封装好的数据结构，要读取其中的数值，需要拆开并转换为np数组
    # kps = np.float32([kp.pt for kp in kps])  # 将kps转换为32位float数组
    return (kps, features)  # 返回特征点集，及其对应的描述特征


# 匹配特征点
def MatchKeyPoints(kps_f, features_f, kps_t, features_t, ratio, reprojThresh):
    # matcher = cv.BFMatcher()  # 使用暴力匹配器
    matcher = cv.FlannBasedMatcher()  # 快速匹配

    # KNN（K-Nearest Neighbor）法即K最邻近法
    # queryDescriptors是要匹配的图像的特征描述子
    # trainDescriptors是用于匹配的参考图像的特征描述子
    # k指定了要返回多少个最近的邻居
    # masks是可选的掩码图像,用于进一步限制匹配的区域。
    # compactResult是一个布尔值,指定是否以紧凑的形式返回匹配点
    # 返回值是一个包含特征匹配结果的数组
    # 特征匹配结果是一个元组，元组内容由k值决定,k=2时，返回一个包含两个DMatch的元组
    # DMatch类型：包含三个非常重要的数据分别是queryIdx，trainIdx，distance
    # queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
    # trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
    # distance：代表这一对匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    knn_neighbor = 2  # 返回几个DMatch数据
    knn_matches = matcher.knnMatch(features_f, features_t, k=knn_neighbor)  # 使用KNN检测sift特征配对

    matches = []
    draw_matches = []
    for m in knn_matches:
        # 当最近距离比次近距离小于ratio时，保留此配对
        if len(m) == knn_neighbor and m[0].distance < m[1].distance * ratio:
            # 存储配对的两个点在数组features_f和数组features_t中的下标
            # 即features_f[m[0].queryIdx] 与 features_t[m[0].trainIdx]特征相似
            matches.append((m[0].queryIdx, m[0].trainIdx))
            draw_matches.append(m)

    # 当筛选后的匹配对大于等于4时，计算视角变换矩阵
    kps_f = np.float32([kp.pt for kp in kps_f])  # 将kps转换为32位float数组
    kps_t = np.float32([kp.pt for kp in kps_t])  # 将kps转换为32位float数组
    if len(matches) >= 4:
        # 获取匹配对点的坐标
        # kps_f[i] 即为第i个特征向量所对应的关键点坐标
        pts_f = np.float32([kps_f[i] for (i, _) in matches])
        pts_t = np.float32([kps_t[i] for (_, i) in matches])

        # 计算视角变换矩阵
        # 要将pts_t的坐标转换为对应的pts_f的坐标，即pts_f = H*pts_t
        H, status = cv.findHomography(pts_t, pts_f, cv.RANSAC, reprojThresh)
        # 返回结果
        return (draw_matches, H, status)


# 拼接函数
def Stitch(images, ratio=0.75, reprojThresh=4.0, drawMatches=False, showMatches=False):
    img_fixed, img_transform = images
    kps_f, features_f = DetetAndDescribe(img_fixed)
    kps_t, features_t = DetetAndDescribe(img_transform)

    # 匹配两张图片的所有特征点，并返回结果
    M = MatchKeyPoints(kps_f, features_f, kps_t, features_t, ratio, reprojThresh)
    if M is not None:  # 如果成功匹配到特征点，结果不为空
        draw_matches, H, status = M
        if drawMatches:  # 如果需要显示特征点匹配效果
            img_darw_matches = cv.drawMatchesKnn(img, kps_f, sti, kps_t, draw_matches[0:10], None, flags=2)
            cv.imshow("DrawMatches", img_darw_matches)
        # 将图片img_transform进行视角变换，返回变换后的结果
        # 参数：原图，变换矩阵，变换后的大小（w,h）
        result = cv.warpPerspective(img_transform, H, (img_fixed.shape[1] + img_transform.shape[1], img_fixed.shape[0]))

        # 将图片A置于图片左侧
        result[0:img_fixed.shape[0], 0:img_fixed.shape[1]] = img_fixed
    if showMatches:
        cv.imshow("Result", result)


if __name__ == '__main__':  # 测试1
    # 使用命令行参数
    arg = ap.ArgumentParser()  # 命令行参数解析器
    arg.add_argument("-i", "--image", required=True, help="path to input image")
    arg.add_argument("-s", "--stitch", required=True, help="path to stitch image")
    args = vars(arg.parse_args())  # 解析参数，获得参数字典
    imgpath = args["image"]  # 分别获得每个参数的数据
    stipath = args["stitch"]

    # 读取图片
    img = cv.imread(imgpath)
    sti = cv.imread(stipath)
    Stitch((img, sti), showMatches=True)  # 图片拼接

    cv.waitKey(0)
