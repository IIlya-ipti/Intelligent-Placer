import cv2
import imageio.v2 as imageio
import src.algorithms as alg
import src.photo_preprocessing as pp

def check_image(path_to_png_jpg_image_on_local_computer):
    ##
    image = imageio.imread(path_to_png_jpg_image_on_local_computer)
    image = cv2.resize(image, (112,208))

    a,b = pp.get_masks(image)

    try:
        polygon = pp.get_poly(image,b[0])
        polygon = polygon.astype("float")
    except Exception as e:
        print("polygon not found")
    
    if len(b) == 1:
        print("without objects!!(canny not detected)")
    for i in b: 
        # 0 is polygon
        if i == 0:continue
        all_valls = pp.get_width_height(b[i])
        wid = all_valls["width"]
        hi = all_valls["height"]
        if alg.add_block(hi,wid,polygon) == False:
            return False
        
    return True