import cv2
import math

img = cv2.imread("./test.jpg")
point_list = []


# 鼠标事件
def MousePoints(event, x, y, flags, param):
    # 检测鼠标左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(point_list) > 0:
            # 利用cv2绘制直线
            cv2.line(img, point_list[0], (x, y), (0, 0, 255), 2)
            # 绘制圆
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        point_list.append((x, y))


# 获取角度
def GetAngle():
    angle = None
    # 获取三个点的未知
    posi0, posi1, posi2 = point_list[0], point_list[1], point_list[2]
    # 计算公式中有除法，所以要处理除法异常
    try:
        tan1 = (posi1[1] - posi0[1]) / (posi1[0] - posi0[0])
    except ZeroDivisionError:  # 如果触发了除0异常，则认为posi1与posi0位于垂直线上
        # 那么夹角应该是90°减去另一条线与水平线的夹角
        angle = round(90 - math.atan(tan2) * 180 / math.pi)
    try:
        tan2 = (posi2[1] - posi0[1]) / (posi2[0] - posi0[0])
    except ZeroDivisionError:
        angle = round(90 - math.atan(tan1) * 180 / math.pi)
    # 未发生除法异常的话angle将未被赋值
    if angle is None:
        angle = round(math.atan((tan2 - tan1) / (1 + tan1 * tan2)) * 180 / math.pi)
    # 将角度的数值绘制到图片上去
    cv2.putText(img, str(angle), point_list[0], cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    point_list.clear()  # 清空点列表
    return angle  # 返回角度


if __name__ == "__main__":

    while True:
        cv2.imshow("img", img)
        # cv2的鼠标回调函数，调用自定义的鼠标事件
        cv2.setMouseCallback("img", MousePoints)
        if len(point_list) == 3:
            GetAngle()

        key = cv2.waitKey(1)
        # 按键q将退出程序
        if key == ord("q"):
            break
        # 按键c将清除图片上的绘制图像
        elif key == ord("c"):
            point_list.clear()
            img = cv2.imread("./test.jpg")
