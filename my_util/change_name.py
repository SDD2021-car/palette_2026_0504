import os


def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if 'ROIs' in filename:
            new_name = filename.split('ROIs', 1)[1]
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))



folder_path = '/data/yjy_data/Palette_S2O_0724/sen1_2/val/B'
rename_files_in_folder(folder_path)
