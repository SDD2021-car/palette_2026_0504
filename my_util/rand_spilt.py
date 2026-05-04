import os
import random
import shutil

list_dir = '/data/ZBC_data/datasets/scenes/train.txt'
list_dir_random = '/data/ZBC_data/datasets/scenes/train_random.txt'
# dir_train = '/data/ZBC_data/datasets/scenes/trainA/train'
dir_test = '/data/ZBC_data/datasets/scenes/test'

if __name__ == '__main__':
    # list_fp = open(list_dir, 'r')
    # list_file = list_fp.readlines()
    # list_file_random = open(os.path.join(list_dir.strip('train.txt'), 'train_random.txt'), 'w')
    # list_idx = [i for i in range(list_file.__len__())]
    # random.shuffle(list_idx)
    # # flag = 1
    # for idx in list_idx:
    #     if list_idx.count(idx)>1:
    #         print('erro')
    #     list_file_random.write(list_file[idx])
    #     # print(flag)
    #     # flag+=1
    # list_file_random.close()
    # list_fp.close()

    list_fp = open(list_dir_random, 'r')
    list_file = list_fp.readlines()
    for file in list_file:
        file = file.rstrip('\n')
        shutil.move(file, dir_test)


