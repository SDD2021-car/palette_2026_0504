import os

# 设置要遍历的文件夹路径
folder_path = '/data/hjf/Palette_S2O/experiments/test_colorization_sar2opt2_240720_161411/results/test/GT'

# 遍历文件夹
for filename in os.listdir(folder_path):
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 确保是文件而不是文件夹
    if os.path.isfile(file_path):
        # 去除文件名前缀两个字母
        new_filename = filename[5:]

        # 构造新的文件路径
        new_file_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(file_path, new_file_path)
        print(f'Renamed "{file_path}" to "{new_file_path}"')

# 注意：请将 '/path/to/your/directory' 替换成实际的文件夹路径