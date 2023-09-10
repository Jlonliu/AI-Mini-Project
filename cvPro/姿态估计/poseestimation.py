import cv2
import mediapipe as mp
import numpy as np
import time


# 姿态识别
class PoseTrack():

    def __init__(self,
                 static=False,
                 complexity=1,
                 smooth=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 detection=0.5,
                 tracking=0.5):
        """
        参数：
            static 表示 静态图像还是连续帧视频
            complexity 表示人体姿态关估计模型；
        0 表示 速度最快，精度最低（三者之中）；
        1 表示 速度中间，精度中间（三者之中）；
        2 表示 速度最慢，精度最高（三者之中）；
            smooth 表示是否平滑关键点；
            enable_segmentation 表示是否对人体进行抠图；
            smooth_segmentation 表示是否对不同输入图像的分割进行滤波以减少抖动
            min_detection_confidence 表示 检测置信度阈值；
            min_tracking_confidence 表示 各帧之间跟踪置信度阈值
        """

        self.static_ = static
        self.smooth_ = smooth
        self.complexity_ = complexity
        self.enable_sqgmentation_ = enable_segmentation
        self.smooth_segmentation_ = smooth_segmentation
        self.detection_ = detection
        self.tracking_ = tracking
        self.mp_pose_ = mp.solutions.pose  # 调用mediapipe库中的hands类
        self.pose_ = self.mp_pose_.Pose(self.static_, self.smooth_, self.complexity_, self.enable_sqgmentation_,
                                        self.smooth_segmentation_, self.detection_, self.tracking_)
        self.mp_draw_ = mp.solutions.drawing_utils  # 用于绘制地标
        self.motion_ = []  # 存储连续几帧的姿态，连成一个动作

    # 查找人体姿态
    def find(self, img, draw=False):
        """
        参数：
            img: 图像
            draw: 是否绘制地标信息，默认为否
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 颜色通道转换
        results = self.pose_.process(img_rgb)  # 处理图像

        if results.pose_landmarks:  # 如果检测到了姿态信息
            # 绘制姿态地标
            if draw:
                self.mp_draw_.draw_landmarks(img, results.pose_landmarks, self.mp_pose_.POSE_CONNECTIONS)

            return results
        else:
            return None

    # 查找姿态地标位置
    def position(self, img, draw=False):
        """
        参数：
            img: 图像
            draw: 是否绘制地标信息，默认为否
        """
        # h, w, c = img.shape  # 获取图片的高,宽,通道
        pose = self.find(img, draw)  # 查找姿态信息并返回
        if pose is not None:  # 如果没有返回空信息
            position_list = []  # 存放地标坐标的列表
            for id, lm in enumerate(pose.pose_landmarks.landmark):  # 遍历姿态地标信息
                # z坐标：离镜头越近z值越小, z值永远小于0，无限远离镜头时，z值趋向于0
                # cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)  # 获取地标坐标
                position_list.append([lm.x, lm.y, lm.z])
            return position_list  # 返回一个33×4维的列表
        else:
            return None

    # 获得连续几帧的姿态地标坐标信息
    def motion(self, img, frame=5, draw=False):
        """
        参数：
            img: 图像
            frame: 一个动作需要几帧图片
            draw: 是否绘制地标信息，默认为否
        """
        # if len(self.motion_) < frame:
        landmark = self.position(img, draw)  # 获取一帧地标信息
        if landmark is not None:  # 如果地标信息非空
            self.motion_.append(landmark)  # 追加一帧地标信息
            if len(self.motion_) == frame:  # 如果帧数够了
                result = np.array(self.motion_)  # 将地标列表转换为np数组
                self.motion_ = []  # 清空姿态地标列表
                return result  # 将地标列表转换为np数组并返回
        else:
            return None

    # 将动作地标信息转换为图片进行显示
    def motion2image(self, motion, show=True):
        """
        参数：
            motion: 动作地标信息
            show: 是否显示转换后的图片
        """
        img = np.array(np.abs(np.array(motion * 255, np.int32)) % 255, np.uint8)
        if show:
            cv2.imshow("MotionImage", img)
        return img


# # 动作识别
# class MotionRec():

#     # 参数，姿态地标列表
#     # 二维列表 33地标×4坐标数据
#     def __init__(self, frame_num=5):
#         self.frame_num_ = frame_num  # 一次处理的帧数
#         self.lm_list_ = []
#         self.counter_ = 0
#         self.ready_ = False  # 地标列表准备好了
#         # self.shake_ = 5  # 姿态抖动幅度，在幅度以内不算改变姿势
#         # self.lm_name_ = mp.solutions.pose.PoseLandmark  # 地标名称
#         # self.motion_ = None     # 当前运动状态

#     def add_landmark(self, landmark):
#         self.lm_list_.append(landmark)  # 追加一层地标
#         self.counter_ += 1  # 计数+1
#         if self.counter_ >= self.frame_num_:  # 如果地标层数超过了阈值
#             self.sift_ = True  # 标识设置为真
#             self.counter_ = 0  # 计数清0


# 运动筛选
class MotionSift():

    # 参数，姿态地标列表
    # 二维列表 33地标×4坐标数据
    def __init__(self, frame_num=5):
        self.frame_num_ = frame_num  # 一次处理的帧数
        self.lm_list_ = []
        self.counter_ = 0
        self.sift_ = False  # 允许运动筛选计算的标识
        self.shake_ = 5  # 姿态抖动幅度，在幅度以内不算改变姿势
        self.lm_name_ = mp.solutions.pose.PoseLandmark  # 地标名称
        self.motion_ = None  # 当前运动状态

    def add_landmark(self, landmark):
        self.lm_list_.append(landmark)  # 追加一层地标
        self.counter_ += 1  # 计数+1
        if self.counter_ >= self.frame_num_:  # 如果地标层数超过了阈值
            self.sift_ = True  # 标识设置为真
            self.counter_ = 0  # 计数清0

    # 右手正在举起
    def right_hand_upping(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cy_rw1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_WRIST][2]  # 获取前一帧的右手腕y坐标
            cy_rw2 = self.lm_list_[i][self.lm_name_.RIGHT_WRIST][2]  # 获取后一帧的右手腕y坐标
            cy_re1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_ELBOW][2]  # 获取前一帧的右手肘y坐标
            cy_re2 = self.lm_list_[i][self.lm_name_.RIGHT_ELBOW][2]  # 获取后一帧的右手肘y坐标
            if cy_rw2 < cy_rw1 or cy_re2 < cy_re1:  # 如果下一帧手腕的y坐标提高了(y坐标越高，值越小)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两的差值帧都满足要求，那么说明手臂在上升
            # 最后一帧的手腕和手肘的y坐标
            last_rw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_WRIST][2]
            last_re_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_ELBOW][2]
            if last_rw_y < last_re_y:  # 如果最后一帧手腕比手肘要高
                # 那么说明右手臂确实在上升
                self.motion_ = "右手举起中"

    # 右手正在放下
    def right_hand_downing(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cy_rw1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_WRIST][2]  # 获取前一帧的右手腕y坐标
            cy_rw2 = self.lm_list_[i][self.lm_name_.RIGHT_WRIST][2]  # 获取后一帧的右手腕y坐标
            cy_re1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_ELBOW][2]  # 获取前一帧的右手肘y坐标
            cy_re2 = self.lm_list_[i][self.lm_name_.RIGHT_ELBOW][2]  # 获取后一帧的右手肘y坐标
            if cy_rw2 > cy_rw1 or cy_re2 > cy_re1:  # 如果下一帧手腕的y坐标降低了(y坐标越低，值越大)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两的差值帧都满足要求，那么说明手臂在下降
            # 最后一帧的手腕和手肘的y坐标
            last_rw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_WRIST][2]
            last_re_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_ELBOW][2]
            if last_rw_y > last_re_y:  # 如果最后一帧手腕比手肘要低
                # 那么说明右手臂确实在下降
                self.motion_ = "右手放下中"

    # 右手向左挥动
    def right_hand_going_left(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cx_rw1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_WRIST][1]  # 获取前一帧的右手腕x坐标
            cx_rw2 = self.lm_list_[i][self.lm_name_.RIGHT_WRIST][1]  # 获取后一帧的右手腕x坐标
            cx_re1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_ELBOW][1]  # 获取前一帧的右手肘x坐标
            cx_re2 = self.lm_list_[i][self.lm_name_.RIGHT_ELBOW][1]  # 获取后一帧的右手肘x坐标
            if cx_rw2 > cx_rw1 or cx_re2 > cx_re1:  # 如果下一帧手腕的x坐标向左移动了(x坐标越向左，值越大)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两帧的差值都满足要求，那么说明手臂在向左移动
            # 最后一帧的手腕和手肘的x坐标
            last_rw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_WRIST][2]
            last_re_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_ELBOW][2]
            if last_rw_y > last_re_y:  # 如果最后一帧手腕比手肘要向左
                # 那么说明右手臂确实在向左移动
                self.motion_ = "右手左移中"

    # 右手向右挥动
    def right_hand_going_right(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cx_rw1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_WRIST][1]  # 获取前一帧的右手腕x坐标
            cx_rw2 = self.lm_list_[i][self.lm_name_.RIGHT_WRIST][1]  # 获取后一帧的右手腕x坐标
            cx_re1 = self.lm_list_[i - 1][self.lm_name_.RIGHT_ELBOW][1]  # 获取前一帧的右手肘x坐标
            cx_re2 = self.lm_list_[i][self.lm_name_.RIGHT_ELBOW][1]  # 获取后一帧的右手肘x坐标
            if cx_rw2 < cx_rw1 or cx_re2 < cx_re1:  # 如果下一帧手腕的x坐标向左移动了(x坐标越向右，值越小)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两帧的差值都满足要求，那么说明手臂在向右移动
            # 最后一帧的手腕和手肘的x坐标
            last_rw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_WRIST][2]
            last_re_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.RIGHT_ELBOW][2]
            if last_rw_y < last_re_y:  # 如果最后一帧手腕比手肘要向右
                # 那么说明右手臂确实在向左移动
                self.motion_ = "右手右移中"

    # 左手正在举起
    def left_hand_upping(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cy_lw1 = self.lm_list_[i - 1][self.lm_name_.LEFT_WRIST][2]  # 获取前一帧的左手腕y坐标
            cy_lw2 = self.lm_list_[i][self.lm_name_.LEFT_WRIST][2]  # 获取后一帧的左手腕y坐标
            cy_le1 = self.lm_list_[i - 1][self.lm_name_.LEFT_ELBOW][2]  # 获取前一帧的左手肘y坐标
            cy_le2 = self.lm_list_[i][self.lm_name_.LEFT_ELBOW][2]  # 获取后一帧的左手肘y坐标
            if cy_lw2 < cy_lw1 or cy_le2 < cy_le1:  # 如果下一帧手腕的y坐标提高了(y坐标越高，值越小)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两帧的差值都满足要求，那么说明手臂在上升
            # 最后一帧的手腕和手肘的y坐标
            last_lw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_WRIST][2]
            last_le_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_ELBOW][2]
            if last_lw_y < last_le_y:  # 如果最后一帧手腕比手肘要高
                # 那么说明右手臂确实在上升
                self.motion_ = "左手举起中"

    # 左手正在放下
    def left_hand_downing(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cy_lw1 = self.lm_list_[i - 1][self.lm_name_.LEFT_WRIST][2]  # 获取前一帧的左手腕y坐标
            cy_lw2 = self.lm_list_[i][self.lm_name_.LEFT_WRIST][2]  # 获取后一帧的左手腕y坐标
            cy_le1 = self.lm_list_[i - 1][self.lm_name_.LEFT_ELBOW][2]  # 获取前一帧的左手肘y坐标
            cy_le2 = self.lm_list_[i][self.lm_name_.LEFT_ELBOW][2]  # 获取后一帧的左手肘y坐标
            if cy_lw2 > cy_lw1 or cy_le2 > cy_le1:  # 如果下一帧手腕的y坐标提高了(y坐标越高，值越小)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两帧的差值都满足要求，那么说明手臂在下降
            # 最后一帧的手腕和手肘的y坐标
            last_lw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_WRIST][2]
            last_le_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_ELBOW][2]
            if last_lw_y > last_le_y:  # 如果最后一帧手腕比手肘要低
                # 那么说明左手臂确实在下降
                self.motion_ = "左手放下中"

    # 左手向左挥动
    def left_hand_going_left(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cx_rw1 = self.lm_list_[i - 1][self.lm_name_.LEFT_WRIST][1]  # 获取前一帧的左手腕x坐标
            cx_rw2 = self.lm_list_[i][self.lm_name_.LEFT_WRIST][1]  # 获取后一帧的左手腕x坐标
            cx_re1 = self.lm_list_[i - 1][self.lm_name_.LEFT_ELBOW][1]  # 获取前一帧的左手肘x坐标
            cx_re2 = self.lm_list_[i][self.lm_name_.LEFT_ELBOW][1]  # 获取后一帧的左手肘x坐标
            if cx_rw2 > cx_rw1 or cx_re2 > cx_re1:  # 如果下一帧手腕的x坐标向左移动了(x坐标越向左，值越大)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两帧的差值都满足要求，那么说明手臂在向左移动
            # 最后一帧的手腕和手肘的x坐标
            last_rw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_WRIST][2]
            last_re_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_ELBOW][2]
            if last_rw_y > last_re_y:  # 如果最后一帧手腕比手肘要向左
                # 那么说明左手臂确实在向左移动
                self.motion_ = "左手左移中"

    # 左手向右挥动
    def left_hand_going_right(self):
        counter = 0
        for i in range(1, len(self.lm_list_)):  # 遍历每一帧
            cx_rw1 = self.lm_list_[i - 1][self.lm_name_.LEFT_WRIST][1]  # 获取前一帧的左手腕x坐标
            cx_rw2 = self.lm_list_[i][self.lm_name_.LEFT_WRIST][1]  # 获取后一帧的左手腕x坐标
            cx_re1 = self.lm_list_[i - 1][self.lm_name_.LEFT_ELBOW][1]  # 获取前一帧的左手肘x坐标
            cx_re2 = self.lm_list_[i][self.lm_name_.LEFT_ELBOW][1]  # 获取后一帧的左手肘x坐标
            if cx_rw2 < cx_rw1 or cx_re2 < cx_re1:  # 如果下一帧手腕的x坐标向左移动了(x坐标越向右，值越小)
                counter += 1  # 每有一帧满足要求，计数器加1

        if counter == self.frame_num_ - 1:  # 如果每两帧的差值都满足要求，那么说明手臂在向右移动
            # 最后一帧的手腕和手肘的x坐标
            last_rw_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_WRIST][2]
            last_re_y = self.lm_list_[self.frame_num_ - 1][self.lm_name_.LEFT_ELBOW][2]
            if last_rw_y < last_re_y:  # 如果最后一帧手腕比手肘要向右
                # 那么说明左手臂确实在向右移动
                self.motion_ = "左手右移中"

    # 运动筛选
    def motion_sift(self):
        if self.sift_:  # 如果允许计算
            self.right_hand_upping()  # 计算。。。
            self.left_hand_upping()
            self.right_hand_downing()
            self.left_hand_downing()
            self.right_hand_going_left()
            self.right_hand_going_right()
            self.left_hand_going_left()
            self.left_hand_going_right()
            self.sift_ = False  # 计算完成后标识设置为假
            self.lm_list_ = []  # 计算完成后将地表列表清空
        else:  # 如果不允许计算
            pass  # 什么都不做

    def show_motion(self):
        if self.motion_ is not None:
            print(self.motion_)


if __name__ == "__main__":
    run_t = time.time()  # 程序开始执行的时间
    c_time, p_time = 0, 0

    mypose = PoseTrack()
    cap = cv2.VideoCapture(0)  # 获取0号摄像头图像

    while True:
        new_t = time.time()  # 获取程序运行的实时时间
        if new_t >= run_t + 20:  # 程序运行n秒后
            break  # 退出程序

        success, img = cap.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        motion = mypose.motion(img, 5, True)
        if motion is not None:
            mypose.motion2image(motion, True)

        # if plist is not None:
        # mysift.motion_sift()
        # mysift.show_motion()

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            print("按下了s")

    # 每一帧图片返回一个姿态列表，根据姿态列表中点的相对位置就可以估计当前姿势
    # 存储三张姿态后进行比较，可以确定姿态的运动趋势
    # 姿态一共33个点
    # 姿态地标点的类
    # lm = mp.solutions.pose.PoseLandmark
    # print(lm.NOSE)
