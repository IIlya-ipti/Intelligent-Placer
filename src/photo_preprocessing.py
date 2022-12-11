import cv2
import cv2 as cv
import numpy as np
from keras.models import load_model
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import sobel
from skimage.measure import label as sk_measure_label
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_erosion
from skimage.segmentation import watershed



    
    
#нас интересует наибольшая площадь объекта
def get_largest_component(mask,arg=-1, ind_of_plygon=0):
    labels = sk_measure_label(mask) # разбиение маски на компоненты связности
    props = regionprops(labels) # нахождение свойств каждой области (положение центра, площадь, bbox, интервал интенсивностей и т.д.)
    areas = [prop.area for prop in props] # нас интересуют площади компонент связности

    #print("Значения площади для каждой компоненты связности: {}".format(areas))
    if arg == -1:
        largest_comp_id = np.array(areas).argmax() # находим номер компоненты с максимальной площадью
        # print([(i,areas[i]) for i in range(len(areas))])
    else:
        largest_comp_id = arg

    #print("labels - матрица, заполненная индексами компонент связности со значениями из множества: {}".format(np.unique(labels)))
    return (labels == (largest_comp_id + ind_of_plygon + 1),areas) # области нумеруются с 1, поэтому надо прибавить 1 к индексу

    
def get_mask_object(rgb_image, ind_of_plygon = 0, largest=True):
    
    
    easy_to_segment = rgb2gray(rgb_image)
    matchbox = easy_to_segment
    canny_edge_map = binary_closing(canny(matchbox, sigma=0.7), footprint=np.ones((4, 4)))
    markers = np.zeros_like(matchbox)  # создаём матрицу markers того же размера и типа, как matchbox
    markers[25:35, 3:7] = 1 # ставим маркеры фона
    markers[binary_erosion(canny_edge_map) > 0] = 2 # ставим маркеры объекта - точки, находящиеся заведомо внутри
    
    sobel_gradient = sobel(matchbox)
    matchbox_region_segmentation = watershed(sobel_gradient, markers)
    
    if largest:
        return get_largest_component(matchbox_region_segmentation,ind_of_plygon)
    return matchbox_region_segmentation

def get_random_photo(photo,mask, left = 0, right = 255):
    # get random photo wighout mask
    photo_copy = np.random.randint(left, right, (photo.shape[0], photo.shape[1],photo.shape[2]))
    mask_1 = mask == False
    mask_2 = mask == True
    mask_1 = mask_1.astype(np.int32)
    mask_2 = mask_2.astype(np.int32)
    first = mask_1.reshape(photo_copy.shape[0],photo_copy.shape[1],1) * photo_copy
    second = mask_2.reshape(photo_copy.shape[0],photo_copy.shape[1],1) * photo
    return first + second

###
model = load_model("test_model.h5")


def get_masks(photo):
    
    # 0 - polygon
    v = model.predict(photo.reshape(1,208,112,3))
    LOW_P = 0.2
    MIN_P = 0.2
    i = 0
    n = get_mask_object(photo,largest=True, ind_of_plygon=0)[1]
    p = []
    masks = {}
    for j in range(1,len(n)):
        if n[j] > 100:
            mask = get_mask_object(photo,largest=True, ind_of_plygon=j)[0]
            #rint("j == ",j)
            for i in range(1,11):
                k = mask * v[0,:,:,i].reshape(208,112)
                k = k > LOW_P
                p_var = sum(k[k != False])/len(mask[mask != False])
                #print(p_var)
                if p_var > MIN_P:
                    p.append(i)
                    masks[i] = mask
                    break
            else:
                masks[0] = mask
                p.append(0)

    return p,masks

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def get_poly(photo, numpy_poly):
    # func to get np.array with polygon
    
    ph = photo.copy()
    
    # get numpy polygon
    image_8bit = np.uint8(numpy_poly * 255)
    contours, _ = cv.findContours(image_8bit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    k = cv2.fillPoly(ph, pts =[box], color=(0,0,0))
    
    
    ##
    _poly_mask = np.ones((k.shape[0],k.shape[1]))
    _poly_null = ~(_poly_mask * k[:,:,0]).astype(bool)
    l = (_poly_null.reshape(_poly_null.shape[0],_poly_null.shape[1],1) * photo)
    
    ##
    to_dall = numpy_poly.reshape(208,112,1) * photo
    to_dall = to_dall > 0
    
    
    
    ##
    l =l *to_dall
    
    ## delete zero vals
    v = np.nonzero(l)
    x = v[0]
    y = v[1]
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    l = l[xl:xr+1, yl:yr+1]
    l = rotate_image(l,90 -  rect[2])
    l = l[:,:,0]
    return l > 0

def get_width_height(numpy_poly):
    
    image_8bit = np.uint8(numpy_poly * 255)
    contours, _ = cv.findContours(image_8bit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    #box = np.int0(box)
    width =int(( (box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)**0.5);
    height = int(( (box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)**0.5);
    return {"width" : width, "height" : height}