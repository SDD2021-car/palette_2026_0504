"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM)
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from metric.imqual_utils import getSSIM, getPSNR, getRMSE, getFSIM
import lpips
# from IPython import embed
# import cv2
import ssim
import torch
from skimage.metrics import structural_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_sort_key(path):
    # 获取文件名
    filename = os.path.basename(path)
    # 返回一个元组，其中第一个元素是文件名（用于字母顺序排序），第二个元素是文件名的长度（用于按长度排序）
    return (filename, len(filename))


## compares avg ssim and psnr
def SSIMs_PSNRs(im_res=(512, 512)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated HR
    """

    dir2 = '/data/cyh/CUT_new/results/sen12_season_2025.04.06/test_400/images/fake_B'
    # dir1 = '/data/yjy_data/Parallel-GAN-main/Parallel-GAN-main/datasets/sar2opt/testB'  # sar2opt jpg
    # dir1 = '/data/yjy_data/pix2pix_and_cyclegan_0808/result_SAR2OPT/s_cyclegan_bs2/experiment_scyc_SAR2OPT_bs2/test_200/realB'  # sar2opt png
    dir1 = '/data/yjy_data/eval/DDBM/GT'    # sen12-season
    # dir1 = '/data/hjf/Dataset/SEN12_Scene/testB'   # sen12-scene


    gtr_paths = sorted(glob(os.path.join(dir1, "*.*")), key=custom_sort_key)
    gen_paths = sorted(glob(os.path.join(dir2, "*.*")), key=custom_sort_key)
    ssims, psnrs, rmses, cw_ssims, lpips_result, fsims = [], [], [], [], [], []
    a = 0
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        print(a)
        a = a + 1
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]

        # read HR from two datasets
        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)
        #
        # 转为张量
        ex_ref = lpips.im2tensor(lpips.load_image(gtr_path))
        ex_p0 = lpips.im2tensor(lpips.load_image(gen_path))
        ex_ref = ex_ref.cuda()
        ex_p0 = ex_p0.cuda()
        ex_d0 = loss_fn.forward(ex_ref, ex_p0)
        ex_d0 = ex_d0.cpu()
        ex_d0_value = ex_d0.detach().numpy()
        lpips_result.append(ex_d0_value.mean())

        s = ssim.SSIM(r_im)
        cw_ssims.append(s.cw_ssim_value(g_im))
        if cw_ssims[-1] < 0.5:
            print("cw_ssim<0.5:cw_ssim=", cw_ssims)
            print(gen_path)

        # get ssim on RGB channels
        # 计算SSIM，返回值越接近1表示两幅图像越相似
        ssim1 = structural_similarity(np.array(r_im), np.array(g_im), data_range=255, channel_axis=2)
        # if ssim1 < 0:
        #     print("ssim<0:ssim=", ssim1)
        #     print(gen_path)
        # ssim1 = getSSIM(np.array(r_im), np.array(g_im))  不需要的代码
        ssims.append(ssim1)

        fsim = getFSIM(np.array(r_im), np.array(g_im))
        fsims.append(fsim)

        rmse = getRMSE(np.array(r_im), np.array(g_im))
        rmses.append(rmse)

        # get psnt on L channel (SOTA norm)
        r_im = r_im.convert("L");
        g_im = g_im.convert("L")
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        if psnr < 20:
            print("psnr<20:psnr=", psnr)
            print(gen_path)
        psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs), np.array(rmses), np.array(cw_ssims), np.array(lpips_result), np.array(fsims)
    # return np.array(rmses),np.array(lpips_result),np.array(fsims)
    # return np.array(ssims)

# """
# Get datasets from
#  - http://irvlab.cs.umn.edu/resources/euvp-dataset
#  - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
# """
# dir = "/data/yjy_data/eval/DDBM"
spatial = True

loss_fn = lpips.LPIPS(net='alex', spatial=spatial)
loss_fn.cuda()
dummy_im0 = torch.zeros(1, 3, 256, 256)  # image should be RGB, normalized to [-1,1]
dummy_im1 = torch.zeros(1, 3, 256, 256)
dummy_im0 = dummy_im0.cuda()
dummy_im1 = dummy_im1.cuda()
dist = loss_fn.forward(dummy_im0, dummy_im1)

### compute SSIM and PSNR
SSIM_measures, PSNR_measures, RMSE_measures, cw_SSIM_measures, LPIPS_measures, FSIM_measures = SSIMs_PSNRs()
# RMSE_measures,LPIPS_measures,FSIM_measures = SSIMs_PSNRs(dir)
# SSIM_measures = SSIMs_PSNRs(dir)
print("SSIM on {0} samples".format(len(SSIM_measures)))
print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
print("MAX: {0} MIN: {1}".format(np.max(SSIM_measures), np.min(SSIM_measures)))
print("PSNR on {0} samples".format(len(PSNR_measures)))
print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))
print("MAX: {0} MIN: {1}".format(np.max(PSNR_measures), np.min(PSNR_measures)))
print("RMSE on {0} samples".format(len(RMSE_measures)))
print("Mean: {0} std: {1}".format(np.mean(RMSE_measures), np.std(RMSE_measures)))
print("MAX: {0} MIN: {1}".format(np.max(RMSE_measures), np.min(RMSE_measures)))
print("CW-SSIM on {0} samples".format(len(cw_SSIM_measures)))
print("Mean: {0} std: {1}".format(np.mean(cw_SSIM_measures), np.std(cw_SSIM_measures)))
print("MAX: {0} MIN: {1}".format(np.max(cw_SSIM_measures), np.min(cw_SSIM_measures)))
print("LPIPS on {0} samples".format(len(LPIPS_measures)))
print("Mean: {0} std: {1}".format(np.mean(LPIPS_measures), np.std(LPIPS_measures)))
print("FSIM on {0} samples".format(len(FSIM_measures)))
print("Mean: {0} std: {1}".format(np.mean(FSIM_measures), np.std(FSIM_measures)))
print("MAX: {0} MIN: {1}".format(np.max(FSIM_measures), np.min(FSIM_measures)))
## Example usage with images
