import os

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if 'ROIs' in filename:
            new_name = filename.split('ROIs', 1)[1]
            # new_name = 'ROIs' + new_name  # 保留“ROIs”
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
            print(f'Renamed: {filename} -> {new_name}')
        else:
            print(f'Skipped: {filename} (No "ROIs" found)')

folder_path = '/data/yjy_data/palette_color/sen_data_new2/train/B'  # 替换为你的文件夹路径
rename_files_in_folder(folder_path)