import cv2

def apply_clahe(img, clip_limit=2.0, tile_size=(8, 8)):
    if img is None:
        return None 

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_optimized = clahe_obj.apply(l)
        lab_optimized = cv2.merge((l_optimized, a, b))
        result = cv2.cvtColor(lab_optimized, cv2.COLOR_LAB2BGR)
    else:
        result = clahe_obj.apply(img)

    return result