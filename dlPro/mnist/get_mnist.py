import pickle
import numpy as np
import cv2


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """

    with open("./mnist.pkl", 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def img_show(name, img):
    # pil_img = Image.fromarray(np.uint8(img))
    # pil_img.show()
    cv2.imshow(name, img)
    cv2.waitKey(0)


def mnist_show():
    (x_train, t_train), (x_test, t_test) = small_minist()
    print("x_train.shape: ", x_train.shape)  # 60000×784的二维数组（60000张图片）
    print("t_train.shape: ", t_train.shape)  # 60000个元素的数组，60000个标签
    print("x_test.shape: ", x_test.shape)  # 10000××784的二维数组（10000张图片）
    print("t_test.shape: ", t_test.shape)  # 10000个元素的数组，10000个标签
    # print("x_train[0]: ", x_train[0])   # 一张图片
    img_show("x_train[0]", x_train[0].reshape(28, 28))
    print("t_train[0]: ", t_train[0])  # 5
    # print("x_test[0]: ", x_test[0])     # 一张图片
    img_show("x_test[0]", x_test[0].reshape(28, 28))
    print("t_test[0]: ", t_test[0])  # 7


def small_minist(quality=None):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    if quality is not None:
        small_x_train = x_train[:quality]
        small_t_train = t_train[:quality]
        small_x_test = x_test[:quality]
        small_t_test = t_test[:quality]
    else:
        small_x_train = x_train
        small_t_train = t_train
        small_x_test = x_test
        small_t_test = t_test
    return (small_x_train, small_t_train), (small_x_test, small_t_test)
