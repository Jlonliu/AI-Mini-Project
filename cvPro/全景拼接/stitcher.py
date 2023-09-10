import cv2

if __name__ == '__main__':

    imgs_path = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    imgs = []
    for i in imgs_path:
        img = cv2.imread(i)
        # img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        imgs.append(img)

    if len(imgs) >= 2:  # 验证读取到的图片数量是否大于等于2张

        # 获取cv2中图片拼接函数
        stitcher = cv2.Stitcher_create()
        #进行图片拼接
        (status, result) = stitcher.stitch(imgs)
        # 如果拼接成功
        if status == cv2.STITCHER_OK:
            cv2.imshow('Result', result)
            cv2.waitKey(0)
        else:
            print('Can not stitch images')