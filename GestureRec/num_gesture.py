# coding: utf-8
import sys
import os
import cv2
import time
# import numpy as np

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import cv.gesture_rec as gr  # 自定义的手势识别文件(使用mediapipe)
import dl.dl_abc as dlabc  # 自定义的深度学习文件（使用pytorch）

# 参数设置
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 手势标签列表
landmark_norm = 21  # 采用norm模式的地标数据个数
landmark_xyz = 63  # 采用xyz模式的地标数据个数
hands = 1  # 检测的手的数量
inlen = landmark_xyz * hands
outlen = len(label_list)
savepath = "./dl_data"  # 训练数据保存路径
readpath = savepath + ".pkl"

# 训练过程中需要实时更改的参数
current_label = 4
best_state_dict_path = "./state_dict_step937_loss0.0032707981293872927.pt"

if __name__ == "__main__":
    # 使用命令行参数
    if len(sys.argv) < 2:
        print("Usage: python game_gestrue_2d.py commend")
        print("采集数据 collect 合并采集的数据 merge 训练数据 train 开始应用 use")
        sys.exit(1)

    commend = sys.argv[1]
    c_time, p_time = 0, 0

    cap = cv2.VideoCapture(0)  # 获取0号摄像头图像
    myhand = gr.GestureRec(hands_num=hands)

    if commend == "collect":  # 采集数据
        for i in range(0, 210):  # 循环210次
            success, img = cap.read()
            # 采集200个手势数据，并制作成pkl数据，用于后续的模型训练
            myhand.make_dl_data(img, path=savepath, label=current_label, nums=200)

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
        for i in range(0, outlen):
            label_list.append(i)
        if len(label_list) != 0:
            myhand.merge_pkl_by_label(path=savepath, label=label_list)

    elif commend == "train":  # 训练数据
        dlabc.Train(readpath, 1000, 100, inlen, outlen, 0.005)

    elif commend == "use":  # 使用模型
        model = dlabc.DLabc(inlen, outlen)
        state_dict_path = best_state_dict_path
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
                # print(num)  # 输出识别的手势
                cv2.putText(img, str(int(num)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
    else:
        pass
