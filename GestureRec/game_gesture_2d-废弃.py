# coding: utf-8
"""
此文件废弃
废弃理由：
    虽然能够准确地识别出来手势，但是没有找到有效地控制游戏角色移动的方式
    pyautogui与pyuserinput的长按按键和物理上的长按按键似乎有区别。
    物理上长按移动键角色可以持续保持移动，但是用python长按却无法实现持续移动
    另外还有动作转换为输入延迟的问题，与电脑同时运行游戏与图像检测的性能问题。
    当然在压根无法控制角色移动的问题下，什么性能问题都无关紧要了。反正都不能用！
"""
import sys
import os
import cv2
import time
import pyautogui as pg
import pykeyboard as pk
# import numpy as np
# import argparse as ap

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import cv.gesture_rec as gr
import dl.dl_abc as dlabc
# from PIL import Image

# # 双手手势
# label_dict = {
#     0: "待机: 双手握拳",
#     1: "上键: 左手拇指向上",
#     2: "下键: 左手拇指向下",
#     3: "左键: 左手拇指向左",
#     4: "右键: 左手拇指向右",
#     5: "上摇杆: 左手四指向上",
#     6: "下摇杆: 左手四指向下",
#     7: "左摇杆: 左手四指向左",
#     8: "右摇杆: 左手四指向右",
#     9: "三角键: 右手数字1",
#     10: "正方键: 右手数字2",
#     11: "叉键: 右手数字3",
#     12: "圆键: 右手数字4",
#     13: "确认键: 右手OK手势",
#     14: "地图键: 右手数字8",
#     15: "攻击键: 右手中指",
#     16: "防御键: 右手握拳",
#     17: "左移攻击: 左手拇指向左 右手中指",
#     18: "右移攻击: 左手拇指向右 右手中指",
#     19: "向上攻击: 左手拇指向上 右手中指",
#     20: "向下攻击: 左手拇指向下 右手中指",
#     21: "左移三角: 左手拇指向左 右手数字1",
#     22: "右移三角: 左手拇指向右 右手数字1",
#     23: "向上三角: 左手拇指向上 右手数字1",
#     24: "向下三角: 左手拇指向下 右手数字1",
#     25: "左移四方: 左手拇指向左 右手数字2",
#     26: "右移四方: 左手拇指向右 右手数字2",
#     27: "向上四方: 左手拇指向上 右手数字2",
#     28: "向下四方: 左手拇指向下 右手数字2",
#     29: "左移叉: 左手拇指向左 右手数字3",
#     30: "右移叉: 左手拇指向右 右手数字3",
#     31: "向上叉: 左手拇指向上 右手数字3",
#     32: "向下叉: 左手拇指向下 右手数字3",
#     33: "左移圆: 左手拇指向左 右手数字4",
#     34: "右移圆: 左手拇指向右 右手数字4",
#     35: "向上圆: 左手拇指向上 右手数字4",
#     36: "向下圆: 左手拇指向下 右手数字4"
# }

# 单手手势
label_dict = {
    0: " ",  #"待机: 握拳",
    1: "up",  #"上键: 拇指向上",
    2: "down",  # "下键: 拇指向下",
    3: "left",  # "左键: 拇指向左",
    4: "right",  # "右键: 拇指向右",
    5: "x",  # "攻击: 中指",
    6: "z",  #"跳跃: 食指",
    7: "右跳: 拇指向右+食指",
    8: "左跳: 拇指向左+食指",
    9: "a",  # : 四指向上",
    10: "m",  # : 手势5",
    11: "enter",  # : 手势OK",
    12: "e",  # : 手势6",
    13: "q",  # : 手势7"
}

hands = 1
inlen = 63 * hands
outlen = len(label_dict)
k = pk.PyKeyboard()
# pg.PAUSE = 0.5  # 每个按键间隔0.5秒


def link_keyboard(num):
    """
    pyautogui.press(['left', 'left', 'left', 'left']) # 按下并松开（轻敲）四下左方向键
    pyautogui.keyDown('shift') # 按下`shift`键
    pyautogui.keyUp('shift') # 松开`shift`键
    pyautogui.hotkey('ctrl', 'v') # 组合按键（Ctrl+V），粘贴功能，按下并松开'ctrl'和'v'按键
    
    """

    if num == 0:
        pass
    elif num == 7:
        pass
        # pg.hotkey(label_list[4], label_list[6])  # 组合按键（left+x）
        # pg.keyUp(label_dict[last_num])  # 松开上一个按键
        # pg.keyDown(label_dict[4])  # 按下当前按键
        # pg.keyDown(label_dict[6])  # 按下当前按键
    elif num == 8:
        pass
        # pg.hotkey(label_list[3], label_list[6])  # 组合按键（left+x）
    else:
        pg.press(label_dict[num])
    return num


if __name__ == "__main__":
    # 使用命令行参数
    if len(sys.argv) < 2:
        print("Usage: python game_gestrue_2d.py commend")
        print("采集数据 collect 合并采集的数据 merge 训练数据 train 开始应用 use")
        sys.exit(1)

    commend = sys.argv[1]

    c_time, p_time = 0, 0

    cap = cv2.VideoCapture(0)  # 获取0号摄像头图像
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands()
    # mp_draw = mp.solutions.drawing_utils
    myhand = gr.GestureRec(hands_num=hands)
    # mygesture = Gesture()

    # m = PyMouse()
    # while True:
    label_max = 13
    savepath = "./dl_data"
    readpath = savepath + ".pkl"

    if commend == "collect":  # 采集数据
        for i in range(0, 120):  # 循环101次
            success, img = cap.read()
            myhand.make_dl_data(img, path=savepath, label=label_max, nums=100)

            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    elif commend == "merge":  # 合并数据
        label_list = []
        for i in range(0, label_max + 1):
            label_list.append(i)
        if len(label_list) != 0:
            myhand.merge_pkl_by_label(path=savepath, label=label_list)
    elif commend == "train":  # 训练数据
        dlabc.Train(readpath, 200, 100, inlen, outlen, 0.005)
    elif commend == "use":  # 使用模型

        model = dlabc.DLabc(inlen, outlen)
        state_dict_path = "./state_dict_step194_loss0.0018418402011905397.pt"
        label_list = []
        for i in range(0, outlen):
            label_list.append(i)

        last_num = 0  # 用来存储上一个手势数字，用来释放按键
        while True:
            success, img = cap.read()
            position = myhand.position(img, True, True)
            position_tensor = myhand.position2tensor(position)
            if position_tensor is not None:
                num = dlabc.Recognition_One_By_One(model, position_tensor, state_dict_path, label_list)
                print(label_dict[num])  # 输出识别的手势
                link_keyboard(num)

            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    else:
        pass
        # num = dlabc.Recognition_One_By_One()

        # datum = myhand.make_dl_data(img, mode=0, label=10, nums=100)
        # if datum is not None:
        #     model = dllearn.MyTest(63)
        #     datum = torch.from_numpy(datum)
        #     num = dllearn.Recognition_One_By_One(model, datum, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        #     print(num + 1)
        # if plist is not None:
        #     myhand.make_dl_data(img)
        # mygesture.gesture(plist)
        # gesture = mygesture.show_gesture()
        # if gesture == "握拳":  # 如果手势是握拳
        #     # x_dim, y_dim = m.screen_size()
        #     m.click(797, 883, 1, 1)
        #     # k.type_string('Hello, World!')
        #     # print(m.position())  # 获取当前鼠标指针的坐标

        # h, w, c = img.shape  # 获取图片的高,宽,通道
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # results = hands.process(img_rgb)

        # if results.multi_hand_landmarks:
        #     for hand_lms in results.multi_hand_landmarks:
        #         for id, lm in enumerate(hand_lms.landmark):
        #             cx, cy = int(lm.x * w), int(lm.y * h)
        #             if id == 4:
        #                 cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        #         mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
        # 显示fps
    # with open("./dl_data.pkl", 'rb') as f:
    #     dataset = pickle.load(f)
    # print(len(dataset))
    # print(len(dataset[1][0]))

    # myhand = HandTrack()
    # myhand.merge_pkl_by_label("./dl_data", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
