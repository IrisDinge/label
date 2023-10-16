import os
import numpy as np
from PIL import Image
import imgviz

def get_gray_cls(array_lbl):
    return np.unique(array_lbl)

def array_gray_to_P(cls_gray, array):
    for idx, val in enumerate(cls_gray):
        array[array == val] = idx
    return array

if __name__ == '__main__':
    label_from_PATH = r"D:\Data\targetDataset\label\test_mix"
    label_to_PATH = r"D:\Data\targetDataset\label\test"
    van_file = r'D:\Data\targetDataset\mask.png'
    van_lbl = Image.open(van_file).convert('L')

    array_lbl = np.array(van_lbl)
    cls_gray = get_gray_cls(array_lbl)

    file_list = os.listdir(label_from_PATH)
    if not os.path.isdir(label_to_PATH):
        os.mkdir(label_to_PATH)

    for file_name in file_list:
        file_path = os.path.join(label_from_PATH, file_name)
        orig_lbl = Image.open(file_path)
        if orig_lbl.mode == 'RGB':
            print("!!!!!!!!!!!!!!!!!!!!!!!BASTARD!!!!!!!!!!!!!!!!!!!!!")
        orig_lbl = orig_lbl.convert('L')
        array_gray = np.array(orig_lbl)
        array_gray[array_gray != 0] = 1
        array_P = array_gray_to_P(cls_gray, array_gray)
        label = Image.fromarray(array_P.astype(np.uint8), mode='P')
        colormap = imgviz.label_colormap()
        palette = [[0, 0, 0], [128, 0, 0]]
        label.putpalette(np.array(palette, dtype=np.uint8).flatten())
        label.save(os.path.join(label_to_PATH, file_name), quality=95)
        print("Congrads to", file_name)

