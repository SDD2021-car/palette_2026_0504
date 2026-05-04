import os
import shutil

def copy_and_rename_images(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith("_fake_B.png"):
            new_filename = filename.replace("_fake_B", "")
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, new_filename)
            shutil.copy(source_path, target_path)

# 示例用法
source_folder = "/data0/czh_data/GVAN/results/cr_GVAN0724/test_140/images/fake_B"  # 原始文件夹
target_folder = "/data0/czh_data/GVAN/results/cr_GVAN0724/test_140/images/results"  # 目标文件夹
copy_and_rename_images(source_folder, target_folder)