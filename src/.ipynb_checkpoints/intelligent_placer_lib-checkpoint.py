import cv2
import imageio.v2 as imageio
import src.algorithms as alg
import src.photo_preprocessing as pp
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import numpy as np


def delete_nulls(array):
    """
    remove extra zeros
    """
    non_null_ind = np.nonzero(array)
    first = min(non_null_ind[0]), min(non_null_ind[1]) 
    second = max(non_null_ind[0]), max(non_null_ind[1])
    
    return array[first[0] : second[0],first[1] : second[1]]

def check_image(path_to_png_jpg_image_on_local_computer) -> bool:
    """
    :param path_to_png_jpg_image_on_local_computer: path image
    :return: does the objects fit into the polygon or not
    """
    ##
    image = imageio.imread(path_to_png_jpg_image_on_local_computer)
    image = cv2.resize(image, (112, 208))

    a, b = pp.get_masks(image)

    try:
        polygon = pp.get_poly(image, b[0])
        polygon = polygon.astype("float32")
    except Exception as e:
        print("polygon not found")
    print("number of objects: ",len(b))
    if len(b) == 1:
        print("without objects!!(canny not detected)")
    for i in b:
        # 0 is polygon
        if i == 0: continue
        all_valls = pp.get_width_height(b[i])
        wid = all_valls["width"]
        hi = all_valls["height"]
        
        flag = False
        
        # rotate object 
        for i in range(0,90,20):
            if alg.add_block(hi, wid, polygon):
                flag = True
                break
            polygon = delete_nulls(pp.rotate_image(polygon,20))
        if flag == False:
            return False
    plt.imshow(polygon)
    return True
