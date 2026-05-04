import os

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):

        new_name = 'ROIs' + filename  # 保留“ROIs”
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
        print(f'Renamed: {filename} -> {new_name}')

folder_path = '/data/hjf/Palette_S2O/sen_data/test/B/'  # 替换为你的文件夹路径
rename_files_in_folder(folder_path)