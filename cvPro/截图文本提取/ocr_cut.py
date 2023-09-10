import cv2 as cv
# import numpy as np
import pytesseract as tes  # OCR链接库
import os
# import pyperclip as pp  # 剪切板相关库
# import matplotlib.pyplot as plt

if __name__ == '__main__':  # 测试1
    # 制定截图后图片的存放路径
    img_path = "E:\\test\\"
    for (root, dirs, files) in os.walk(img_path, topdown=False):
        if dirs == []:  # 过滤掉所有文件夹，只读取文件
            files = sorted(files)
            img_name = files[-1]  # 获取最新截图名称
    # # img = cv.imread("./puretextBook.png")
    # # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 获取灰度图
    # print(img_name)
    # # # for i in img_gray:
    # # #     for j in i:
    # # #         print(j)
    # # cv.imshow("PureTextBook", img_gray)

    # # cv.waitKey(0)

    # # if len(sys.argv) < 2:
    # #     print("Usage: python ocr.py image.png")
    # #     sys.exit(1)

    # # 使用命令行参数
    # # impath = sys.argv[1]

    # # -l    选择语言

    # # --oem 使用LSTM作为OCR引擎, 可选值为 0、1、2、3;
    # # 0     Legacy engine only 仅限传统引擎
    # # 1     Neural nets LSTN engine only 仅限神经网络LSTN引擎
    # # 2     Legacy + LSTM engines
    # # 3     Default,based on what is available 默认值，基于可用内容
    # # --psm 设 置 page Segmentation 模式为自动

    config = ("-l chi_sim --oem 1 --psm 3")
    print(img_path + img_name)
    img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)
    # # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # # img_h, img_w, c = img.shape

    # # 进行识别, 本质上是调用tesseract命令行工具
    # # boxes = tes.image_to_boxes(img, config=config)  # 识别图片中的文字，生成文字内容以及所在位置信息
    text = tes.image_to_string(img, config=config)  # 识别图中文字，输出为字符串
    # # for b in boxes.splitlines():
    # #     b = b.split(" ")
    # #     # 分别获得文字位置的左上角坐标以及宽度和高度
    # #     x, y, w, h = int(b[1]), img_h - int(b[2]), int(b[3]), img_h - int(b[4])
    # #     # 在图片中框出文字
    # #     cv.rectangle(img=img, pt1=(x, y), pt2=(w, h), color=(0, 0, 255), thickness=1)
    # #     # 将文字输出为cv2的字体，并显示在图片上
    # #     cv.putText(img, b[0], (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    print(text.replace(" ", ""))

    # # cv.imshow("Result", img)
    # # cv.waitKey(0)
