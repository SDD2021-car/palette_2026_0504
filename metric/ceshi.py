import numpy as np
from PIL import Image
import torch

A = Image.open("/home/czh/Desktop/PITI-palette/test/sar2opt0525-1/GT/1_2640_4800.png")
B = Image.open("/home/czh/Desktop/PITI-palette/test/sar2opt0525-1/HR/1_2640_4800.png")

#a = np.array(A, dtype=np.uint32)
#b = np.array(B, dtype=np.uint32)
#print('a:', a.shape, a)
#print('b:', b.shape, b)

#M = np.multiply(a, b)
#print('M:', M.shape, M)

#a2 =  np.array([[[69, 80, 63]]])
#b2 =  np.array([[[60, 74, 64]]])

#print('a2:', a2.shape, a2)
#print('b2:', b2.shape, b2)

#M2 = np.multiply(a2, b2)
#print('M2:', M2.shape, M2)

# y = np.sum(M, axis=2)
# print('y:', y)



def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_pil(normalize=True, toPIL=True):
    transforms_list = []
    if normalize:
        transforms_list += [transforms.Normalize((-1.0, -1.0, -1.0),
                                                 (2.0, 2.0, 2.0))]
    if toPIL:
        transforms_list += [transforms.ToPILImage()]


