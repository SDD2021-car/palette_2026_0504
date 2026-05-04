import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from PIL import Image
from models.guided_diffusion_modules.g_theta import UNet



def get_pairs(points, get_line_method):
    if get_line_method == 'random':
        np.random.shuffle(points)
        pairs = points.reshape(10, 2, 2)
        return pairs
    elif get_line_method == 'dijkstra_like':
        pass
    elif get_line_method == 'sam_based':
        pass

    # 1. 读取size为256*256的3通道RGB图像
# 这里我们创建一个随机的RGB图像，您也可以使用Image.open()读取实际的图像
image_path = '/data/hjf/Palette_S2O/sen_data/train/B/ROIs1868_summer_s1_38_p1.png'  # 替换为您的图像路径
img = Image.open(image_path).resize((256, 256))
img = np.array(img, dtype=np.float32)
image = np.array(img, dtype=np.float32)

# 读取灰度图像并转换为numpy数组
gray_image_path = '/data/hjf/Palette_S2O/sen_data/train/A/ROIs1868_summer_s1_38_p1.png'  # 替换为您的灰度图像路径
gray_image = Image.open(gray_image_path).resize((256, 256)).convert('L')
gray_image = np.array(gray_image, dtype=np.float32)

# 将灰度图像扩展到3通道
gray_image_3d = np.stack([gray_image]*3, axis=-1)

# 2. 在RGB图像上等概率任取20个点
points = np.column_stack((
    np.random.randint(0, 256, 20),
    np.random.randint(0, 256, 20)
))

get_line_method = 'random'
pairs = get_pairs(points, get_line_method)

# 创建一个掩码，用于标记需要保留的像素点
mask = np.zeros((256, 256), dtype=bool)

# 4. 将配对的两个点之间的连线经过的像素点保留
for pair in pairs:
    rr, cc = line(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
    mask[rr, cc] = True

# 将掩码扩展到3个通道
mask_3d = np.stack([mask]*3, axis=-1)

# 5. 将剩余像素点（不在任意一条连上的像素点）的值设置为空NaN
image[~mask_3d] = 0

# 5. 将保留的连线像素点与灰度图相加
fused_image = np.copy(gray_image_3d)
fused_image[mask_3d] = image[mask_3d]

# 确保像素值在0-255范围内
fused_image = np.clip(fused_image, 0, 255)  # 确保不超过255

# 将像素值归一化到[0, 1]范围以便可视化
fused_image /= 255

# 可视化结果
plt.imshow(fused_image)
plt.axis('off')
plt.show()

g_theta_net = UNet(
    in_channel= 6,
    out_channel=3,
    inner_channel = 64,
    channel_mults = [
            1,
            2,
            4,
            8
        ],
    attn_res = [
      16
        ],
    num_head_channels = 32,
    res_blocks = 2,
    dropout = 0.2,
    image_size = 224
    )
fm1, fm2 = g_theta_net(fused_image)