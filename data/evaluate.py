import torch
import math
import os
import cv2
import numpy as np
from scipy.signal import convolve2d
# import fid_score

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    ssim_total = []
    for i in range(1, 3):
        im1t = im1[:, :, i-1]
        im2t = im2[:, :, i-1]
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
        window = window / np.sum(np.sum(window))

        if im1t.dtype == np.uint8:
            im1t = np.double(im1t)
        if im2t.dtype == np.uint8:
            im2t = np.double(im2t)

        mu1 = filter2(im1t, window, 'valid')
        mu2 = filter2(im2t, window, 'valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = filter2(im1t * im1t, window, 'valid') - mu1_sq
        sigma2_sq = filter2(im2t * im2t, window, 'valid') - mu2_sq
        sigmal2 = filter2(im1t * im2t, window, 'valid') - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_total.append(ssim_map)

    return np.mean(np.mean(ssim_total))


def psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
       return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def cosin(img1, img2):
    [height, width, channel] = img1.shape
    img1vector = np.reshape(img1, height * width * channel)
    img2vector = np.reshape(img2, height * width * channel)

    return torch.cosine_similarity(torch.from_numpy(img1vector*1.0), torch.from_numpy(img2vector*1.0), dim=0)


if __name__ == '__main__':
    img1folder = '/data/hjf/Palette_S2O/experiments/test_colorization_sar2opt2_240603_173758/results/test/0'
    img2folder = '/data/hjf/Palette_S2O/experiments/test_colorization_sar2opt2_240603_173758/results/test/GT'
    psnr_total = []
    ssim_total = []
    # cosin_total = []
    # fid_total = []

    for img in os.listdir(img1folder):
        img1 = cv2.imread(img1folder + '/' + img)
        # img = img.replace("fake", "real")
        img2 = cv2.imread(img2folder + '/' + img)

        psnr_score = psnr(img1, img2)
        ssim_score = ssim(img1, img2)
        # cosin_score = cosin(img1, img2)

        psnr_total.append(psnr_score)
        ssim_total.append(ssim_score)
        # cosin_total.append(cosin_score)
    # fid = fid_score.fid(img1folder, img2folder)

    print('PSNR: ', np.mean(psnr_total))
    print('SSIM: ', np.mean(ssim_total))
    # print('cosin: ', np.mean(cosin_total))
    # print('FID: ', np.mean(fid_total))
