import cv2
import mediapipe as mp
import numpy as np
import time


# 手部检测类
class HandTrack:

    def __init__(self, static=False, hands_num=2, complexity=1, detection=0.5, tracking=0.5):
        """
        参数：
            static: 如果设置为 false,该解决方案会将输入图像视为视频流,
        它将尝试在第一个输入图像中检测手,并在成功检测后进一步定位手的地标。
        在随后的图像中,一旦检测到所有 max_num_hands 手并定位了相应的手的地标,
        它就会简单地跟踪这些地标,而不会调用另一个检测,直到它失去对任何一只手的跟踪。
        这减少了延迟,非常适合处理视频帧。
        如果设置为 true,则在每个输入图像上运行手部检测,非常适合处理一批静态的、可能不相关的图像。
        默认为false
            hands_num: 要检测的最多的手数量。默认为2。
            complexity: 手部模型的复杂性: 0或1,
            detection: 来自手部检测模型的最小置信值 [0.0, 1.0],用于将检测视为成功。默认为 0.5。
            tracking: 来自地标跟踪模型的最小置信值 [0.0, 1.0],
        将其设置为更高的值可以提高解决方案的稳健性,但代价是更高的延迟。
        如果 mode 为True,则忽这个参数略,手部检测将在每个图像上运行
        """

        self.static_ = static
        self.hands_num_ = hands_num
        self.complexity_ = complexity
        self.detection_ = detection
        self.tracking_ = tracking
        self.mp_hands_ = mp.solutions.hands  # 调用mediapipe库中的hands类
        """
        self.mp_hands_.Hands返回值
        MULTI_HAND_LANDMARKS：被检测/跟踪的手的集合,其中每只手被表示为21个手部地标的列表,
        每个地标由x、y和z组成。x和y分别由图像的宽度和高度归一化为[0.0,1.0]。
        Z表示地标深度,以手腕深度为原点,值越小。
        MULTI_HANDEDNESS：被检测/追踪的手是左手还是右手的集合。
        每只手由label （标签）和score （分数）组成。
        label 是“Left”或“Right”值的字符串。
        score 是预测左右手的估计概率。
        """
        self.hands_ = self.mp_hands_.Hands(self.static_, self.hands_num_, self.complexity_, self.detection_, self.tracking_)
        self.mp_draw_ = mp.solutions.drawing_utils  # 用于绘制手部地标
        self.dl_data = []  # 存储训练数据

    # 查找手部
    def find(self, img, draw=False):
        """
        参数：
            img: 图像
            draw: 是否绘制地标信息，默认为否
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 颜色通道转换
        results = self.hands_.process(img_rgb)  # 处理图像

        if results.multi_hand_landmarks:  # 如果检测到了手部信息
            # 绘制手部地标
            if draw:
                for hand_lms in results.multi_hand_landmarks:  # 遍历每一只手
                    # 绘制一只手的地标信息
                    # HAND_CONNECTIONS表示对地标点进行连线
                    self.mp_draw_.draw_landmarks(img, hand_lms, self.mp_hands_.HAND_CONNECTIONS)

            return results
        else:
            return None

    # 查找手部地标位置
    def position(self, img, hand_num=0, relative=False, draw=False):
        """
        参数：
            img: 图像
            hand_num: 查找那一只手的信息, 默认为0号
            relative: 返回的坐标信息是否使用手部其他地标坐标相对于手腕的坐标
            draw: 是否绘制地标信息，默认为否
        """
        # h, w, c = img.shape  # 获取图片的高,宽,通道
        hands = self.find(img, draw)  # 查找手部信息并返回
        if hands is not None:  # 如果没有返回空信息
            position_list = []  # 存放地标坐标的列表
            hand = hands.multi_hand_landmarks[hand_num]  # 获取其中一只手
            # 将字典landmark转换为列表landmark
            for lm in hand.landmark:  # 遍历手部地标信息
                position_list.append([lm.x, lm.y, lm.z])  # 获取绝对地标坐标
                position_arr = np.array(position_list, np.float32)  # 将列表转换为数组
            # 判断以何种形式返回地标坐标信息
            if relative:  # 如果使用相对坐标
                position = np.zeros((len(position_list), len(position_list[0])), np.float32)
                for i in range(0, len(position_list)):
                    # 每个地标都减去0号（手腕）地标，获取相对坐标
                    position[i] = position_arr[i] - position_arr[0]
            else:
                # 获取绝对坐标
                position = position_arr

            return position
        else:
            return None


if __name__ == "__main__":
    c_time, p_time = 0, 0

    cap = cv2.VideoCapture(0)  # 获取0号摄像头图像
    myhand = HandTrack()

    while True:

        success, img = cap.read()
        if success:
            myhand.position(img, draw=True)

        # 显示fps
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break