import cv2
import mediapipe as mp
import numpy as np
# import time
import pickle
# from pymouse import PyMouse
import torch

# 常量
FEATURE_XYZ = 0  # 以地标xyz坐标为特征
FEATURE_NORM = 1  # 以地标距离原点的模长为特征
LANDMARK_NUMS = 21  # 地标数量
DIMENSION = 3  # 坐标维度


# 手部检测类
class GestureRec:

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
    def find(self, img, draw=True):
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
    def position(self, img, relative=True, draw=True):
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
            for hand in hands.multi_hand_landmarks:
                for lm in hand.landmark:  # 遍历手部地标信息，依次获取两只手的地标信息
                    position_list.append([lm.x, lm.y, lm.z])  # 获取绝对地标坐标
            position_arr = np.array(position_list, np.float32)  # 将列表转换为数组

            # 判断以何种形式返回地标坐标信息
            if relative:  # 如果使用相对坐标
                position = np.zeros((LANDMARK_NUMS * self.hands_num_, DIMENSION), np.float32)
                for i in range(0, len(position_list)):
                    # 每个地标都减去0号（手腕）地标，获取相对坐标
                    position[i] = position_arr[i] - position_arr[0]
            else:
                # 获取绝对坐标
                position = np.zeros((LANDMARK_NUMS * self.hands_num_, DIMENSION), np.float32)
                for i in range(0, len(position_list)):
                    position[i] = position_arr[i]

            return position
        else:
            return None

        #     hand = hands.multi_hand_landmarks[hand_num]  # 获取其中一只手
        #     # 将字典landmark转换为列表landmark
        #     for lm in hand.landmark:  # 遍历手部地标信息
        #         position_list.append([lm.x, lm.y, lm.z])  # 获取绝对地标坐标
        #     position_arr = np.array(position_list, np.float32)  # 将列表转换为数组
        #     # 判断以何种形式返回地标坐标信息
        #     if relative:  # 如果使用相对坐标
        #         position = np.zeros((len(position_list), len(position_list[0])), np.float32)
        #         for i in range(0, len(position_list)):
        #             # 每个地标都减去0号（手腕）地标，获取相对坐标
        #             position[i] = position_arr[i] - position_arr[0]
        #     else:
        #         # 获取绝对坐标
        #         position = position_arr

        #     return position
        # else:
        #     return None

    # 将二维数组转换成一维数组
    def position2numpy(self, position, feature=FEATURE_XYZ):
        if position is not None:
            if feature == FEATURE_XYZ:
                return np.array(position).reshape(len(position) * 3, )
            elif feature == FEATURE_NORM:
                results = np.zeros(len(position), np.float32)
                for i in range(0, len(position)):
                    results[i] = np.linalg.norm(position[i])
                return results
            else:
                pass
        else:
            return None

    # 将二维数组转换成一维张量
    def position2tensor(self, position, feature=FEATURE_XYZ):
        # 先转换为1维度数组
        position_numpy = self.position2numpy(position, feature)
        if position_numpy is not None:
            # 再转换为1维张量
            return torch.from_numpy(position_numpy)
        else:
            return None

    # 制作深度学习用的数据
    def make_dl_data(self, img, path="./dl_data", feature=FEATURE_XYZ, relative=True, draw=True, label=None, nums=100):
        """
        参数：
            img: 要检测的图片
            mode: 制作数据的模式
                0: 将xyz坐标以此排列成一个数组(共计21×3个数值)；
                1: 将xyz坐标到0点的长度排成一个列表（共计21个非负数值）
            label: 训练数据的标签
            nums: 一次制作多少个训练数据
        """
        position = self.position(img, relative, draw)
        if position is not None:
            position_numpy = self.position2numpy(position, feature)
            # if mode == 0:
            #     results = self.position2numpy(position)
            # elif mode == 1:
            #     results = np.zeros(len(position), np.float32)
            #     for i in range(0, len(position)):
            #         results[i] = np.linalg.norm(position[i])

            if label is None:
                print("缺少标签，无法生产训练数据")
                print(position_numpy)
            else:
                dl_datum = [position_numpy, label]  # 对应数据和标签
                self.dl_data.append(dl_datum)  # 存入列表
                if len(self.dl_data) >= nums:  # 如果读取到了足够的数据
                    # 生成pickle文件存储数字图像，以便其他python程序调用
                    with open(path + "_" + str(label) + ".pkl", 'wb') as f:
                        pickle.dump(self.dl_data, f, -1)
                        print("Done!")
                    self.dl_data = []  # 清空缓存数据

    # 沿着标签合并pkl文件
    def merge_pkl_by_label(self, path="./dl_data", label=None):
        if label is not None:
            dataset = []
            for i in label:
                with open(path + "_" + str(i) + ".pkl", 'rb') as f:
                    data = pickle.load(f)
                    for j in data:
                        dataset.append(j)

            if len(dataset) != 0:
                # 生成pickle文件存储数字图像，以便其他python程序调用
                with open(path + ".pkl", 'wb') as f:
                    pickle.dump(dataset, f, -1)
                    print("Done!")

    # 打开pkl查看里面的数据
    def load_pkl(self, path, index):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            print(len(data[0][0]))
            for i in data:
                print(i[index])
