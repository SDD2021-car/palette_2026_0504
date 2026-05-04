from PIL import Image
import os

img_dir = '/data/hjf/Palette_S2O/experiments/test_colorization_sar2opt2_240720_161411/results/test/GT'
for filename in os.listdir(img_dir):
    if filename.endswith('jpg') or filename.endswith('png'):
        img_path = os.path.join(img_dir,filename)
        with Image.open(img_path) as img:
            gray_img = img.convert('L')
            gray_img_path = os.path.join(img_dir, 'gray_'+filename)
            gray_img.save(gray_img_path)
