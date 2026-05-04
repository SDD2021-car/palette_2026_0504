import cv2
import numpy as np


def sobel_edge_detection(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算x和y方向的Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 归一化到0-255
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    return magnitude


# 读取图像
image = cv2.imread('C:\Users\I‘m Giant\.cursor-tutor\image_test\ROIs1868_summer_s1_38_p4_fake_A.png')

# 应用Sobel边缘检测
edge_image = sobel_edge_detection(image)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Sobel Edge Detection', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('sobel_edge_image.jpg', edge_image)