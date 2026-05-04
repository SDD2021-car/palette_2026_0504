from skimage.feature import local_binary_pattern
import cv2


def lbp(image, radius=3):

    n_points = 8 * radius
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_result = local_binary_pattern(image, n_points, radius)
    return lbp_result


if __name__ == '__main__':
    image = cv2.imread('/home/zbc/code/model_test/ROIs1868_summer_18_p1_real_A.png')
    result = lbp(image)