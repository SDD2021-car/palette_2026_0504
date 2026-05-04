import os
import shutil

def move_files_from_list(txt_file_path, source_folder, destination_folder):
    # 读取txt文件中的文件名
    with open(txt_file_path, 'r') as file:
        file_names = file.read().splitlines()

    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)

    # 将文件从源文件夹移动到目标文件夹
    for file_name in file_names:
        file_name = os.path.basename(file_name)
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name}")
        else:
            print(f"File not found: {file_name}")

# 使用示例
txt_file_path = '/data/czh/sar2opt/sar2opt_val.txt'  # 指定包含文件名的txt文件路径
source_folder = '/data/czh/sar2opt/train_B'  # 指定源文件夹路径
destination_folder = '/data/czh/sar2opt/val_B'  # 指定目标文件夹路径

move_files_from_list(txt_file_path, source_folder, destination_folder)
