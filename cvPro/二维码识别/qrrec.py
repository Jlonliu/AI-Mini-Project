import cv2
import numpy as np
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(1)  # 读取摄像头


def QRRead(img):
    # 识别图像中的二维码信息并遍历
    # decode(img)函数返回一个包含img中二维码（条形码）信息的列表
    # 信息包括：二维码数据、类型、尺寸、几何坐标，quality,方向
    for barcode in decode(img):
        # 获取二维码图像的角点的几何坐标
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))  # 将角点的几何坐标转换成利于cv2使用的形状
        # 绘制多边形
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        # 绘制文字
        cv2.putText(img, barcode.data.decode('utf-8'), (barcode.rect.left, barcode.rect.top - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        return barcode.data.decode('utf-8')


while True:
    # 读取图像
    success, img = cap.read()
    # 识别图像中的二维码信息
    if success:
        info = QRRead(img)
        if info is not None:
            print(info)

        cv2.imshow('img', img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break