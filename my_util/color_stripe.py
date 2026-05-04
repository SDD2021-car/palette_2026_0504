import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from PIL import Image
import os

def list_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                yield os.path.join(root, file)


# image = np.random.randint(0, 256, (256, 256, 3), dtype=np.float32)
# 指定你的图像文件夹路径
image_folder = '/data/hjf/Palette_S2O/sen_data/train/B'
image_save_folder = '/data/hjf/Palette_S2O/sen_data/train/color_stripe'
for image_path in list_images(image_folder):
    image = Image.open(image_path).resize((256,256))
    image = np.array(image,dtype=np.float32)

# 2. 在RGB图像上等概率任取20个点
    points = np.column_stack((
        np.random.randint(0, 256, 20),
        np.random.randint(0, 256, 20)
    ))

# 3. 将20个点任意两两配对
    np.random.shuffle(points)
    pairs = points.reshape(10, 2, 2)

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

# 可视化结果

    plt.imshow(image/255)  # 将像素值归一化到[0,1]范围
    plt.axis('off')
    plt.show()
    image = Image.fromarray(image)
    image.save('')



