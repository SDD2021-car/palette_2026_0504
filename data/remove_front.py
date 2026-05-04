import os
import random
def extract_filenames(folder_path, output_file):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(file)

    random.shuffle(filenames)

    with open(output_file, 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')
        print(f"文件名提取成功，并已写入到 {output_file} 中。")

# 用法示例
output_file_path = '/data/yjy_data/dataset/SAR2Opt/SAR2Opt_new_val.txt'  # 输出文件路径
folder_path = '/data/yjy_data/dataset/SAR2Opt/val/A'

extract_filenames(folder_path, output_file_path)
