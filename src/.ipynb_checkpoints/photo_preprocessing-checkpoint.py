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


def get_largest_component(mask, arg=-1, ind_of_plygon=0):
    labels = sk_measure_label(mask)  # разбиение маски на компоненты связности
    props = regionprops(
        labels)  # нахождение свойств каждой области (положение центра, площадь, bbox, интервал интенсивностей и т.д.)
    areas = [prop.area for prop in props]  # нас интересуют площади компонент связности

    # print("Значения площади для каждой компоненты связности: {}".format(areas))
    if arg == -1:
        largest_comp_id = np.array(areas).argmax()  # находим номер компоненты с максимальной площадью
        # print([(i,areas[i]) for i in range(len(areas))])
    else:
        largest_comp_id = arg

    # print("labels - матрица, заполненная индексами компонент связности со значениями из множества: {}".format(np.unique(labels)))
    return (labels == (largest_comp_id + ind_of_plygon + 1),
            areas)  # области нумеруются с 1, поэтому надо прибавить 1 к индексу


def get_mask_object(rgb_image, ind_of_polygon=0, largest=True):
    """
    :param rgb_image: numpy array (None,None,None,3)
    :param ind_of_polygon: ind polygon of all polygons in image
    :param largest: return ind of polygons or not
    :return: masks (polygons)
    """
    easy_to_segment = rgb2gray(rgb_image)
    matchbox = easy_to_segment
    canny_edge_map = binary_closing(canny(matchbox, sigma=0.7), footprint=np.ones((4, 4)))
    markers = np.zeros_like(matchbox)  # создаём матрицу markers того же размера и типа, как matchbox
    markers[25:35, 3:7] = 1  # ставим маркеры фона
    markers[binary_erosion(canny_edge_map) > 0] = 2  # ставим маркеры объекта - точки, находящиеся заведомо внутри

    sobel_gradient = sobel(matchbox)
    matchbox_region_segmentation = watershed(sobel_gradient, markers)

    if largest:
        return get_largest_component(matchbox_region_segmentation, ind_of_polygon)
    return matchbox_region_segmentation


def get_random_photo(photo, mask, left=0, right=255):
    """
    :param photo: numpy array
    :param mask: numpy array mask
    :param left: min color
    :param right: max color
    :return: photo with background change
    """
    # get random photo wighout mask
    photo_copy = np.random.randint(left, right, (photo.shape[0], photo.shape[1], photo.shape[2]))
    mask_1 = mask == False
    mask_2 = mask == True
    mask_1 = mask_1.astype(np.int32)
    mask_2 = mask_2.astype(np.int32)
    first = mask_1.reshape(photo_copy.shape[0], photo_copy.shape[1], 1) * photo_copy
    second = mask_2.reshape(photo_copy.shape[0], photo_copy.shape[1], 1) * photo
    return first + second


###
model = load_model("test_model.h5")


def get_masks(photo):
    """
    :param photo: photo
    :return: probability of belonging to a class and classes masks
    """
    # 0 - polygon
    predict_mask = model.predict(photo.reshape(1, 208, 112, 3))
    LOW_P_FOR_MODEL_CLASSIFICATION = 0.2
    MIN_P_FOR_CLASSIFICATION = 0.2
    polygons = get_mask_object(photo, largest=True, ind_of_polygon=0)[1]
    probability_of_class = []
    masks_not_null = list()
    masks = {}
    
    for j in range(1, len(polygons)):
        if polygons[j] > 100:
            mask = get_mask_object(photo, largest=True, ind_of_polygon=j)[0]
            for i in range(1, 11):
                k = mask * predict_mask[0, :, :, i].reshape(208, 112)
                k = k > LOW_P_FOR_MODEL_CLASSIFICATION
                p_var = sum(k[k != False]) / len(mask[mask != False])
                # print(p_var)
                if p_var > MIN_P_FOR_CLASSIFICATION:
                    probability_of_class.append(i)
                    masks_not_null.append((i,mask))
                    break
            else:
                masks[0] = mask
                probability_of_class.append(0)
    return probability_of_class, masks_not_null, masks

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    
    # to ones
   # rotated_mat = rotated_mat.where(rotated_mat > 0.4, 1)
    #plt.imshow(rotated_mat)
    
    return rotated_mat




def get_poly(photo, numpy_poly):
    """
    :param photo: photo
    :param numpy_poly: get numpy array of polygon
    :return: numpy min size rectangle array
    """
    # func to get np.array with polygon

    ph = photo.copy()

    # get numpy polygon
    image_8bit = np.uint8(numpy_poly * 255)
    contours, _ = cv.findContours(image_8bit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    k = cv2.fillPoly(ph, pts=[box], color=(0, 0, 0))

    ##
    _poly_mask = np.ones((k.shape[0], k.shape[1]))
    _poly_null = ~(_poly_mask * k[:, :, 0]).astype(bool)
    l = (_poly_null.reshape(_poly_null.shape[0], _poly_null.shape[1], 1) * photo)

    ##
    to_dall = numpy_poly.reshape(208, 112, 1) * photo
    to_dall = to_dall > 0

    ##
    l = l * to_dall

    ## delete zero vals
    v = np.nonzero(l)
    x = v[0]
    y = v[1]
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    l = l[xl:xr + 1, yl:yr + 1]
    l = rotate_image(l, 90 - rect[2])
    l = l[:, :, 0]
    return l > 0


def get_width_height(numpy_poly):
    """
    :param numpy_poly: numpy polygon
    :return: width and height of numpy_poly
    """
    image_8bit = np.uint8(numpy_poly * 255)
    contours, _ = cv.findContours(image_8bit, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    # box = np.int0(box)
    width = int(((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2) ** 0.5);
    height = int(((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2) ** 0.5);
    return {"width": width, "height": height}
