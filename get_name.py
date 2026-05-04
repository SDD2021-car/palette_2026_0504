import os

def get_image_files(directory):
    # 定义支持的图片文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = []
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为支持的图片格式
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

def write_image_names_to_txt(image_files, output_file):
    # 按文件名排序
    image_files.sort()
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_file in image_files:
            # 只写入文件名，不包含路径
            f.write(os.path.basename(image_file) + '\n')

if __name__ == "__main__":
    # 指定要遍历的目录
    directory = '/data/cyh/Datasets/GF3_sar2opt/testA'  # 当前目录
    # 指定输出的文本文件
    output_file = '/data/cyh/Datasets/GF3_sar2opt/GF3_sar2opt_test.txt'


    # 获取所有图片文件
    image_files = get_image_files(directory)
    # 将图片文件名写入文本文件
    write_image_names_to_txt(image_files, output_file)
    print(f"图片文件名已按顺序写入 {output_file}")
