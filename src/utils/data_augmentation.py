import cv2
from datetime import datetime
from utils.data import get_random_voc_img
from utils.draw import filter_color
from skimage.measure import regionprops
import numpy as np


def crop(img, bbox):
    """Crop image using bbox (y1, x1, y2, x2)"""
    return img[bbox[0]:bbox[2], bbox[1]:bbox[3]]


def crop_resize(img, bbox, dsize):
    """Crop and resize image using bbox (y1, x1, y2, x2) and dsize (width, height)"""
    return cv2.resize(crop(img, bbox), dsize=dsize)


def log(log):
    """Print log with timestamp"""
    print(datetime.now().strftime("%H:%M:%S.%f")[:-3], log)
    



def voc_data_augmentation(tar_img, tar_seg, voc_img_dir, voc_seg_dir, cover_factor=0.8):
    """Data augmentation using VOC dataset"""
    
    tar_shape = tar_seg.shape    
    log("Get Fashion image/segmentation, shape: "+ str(tar_shape))


    # ====================================================================================================
    # 1. Select random VOC IMG and segmentation
    # voc_img, voc_seg, voc_id = get_voc_img(voc_bus, voc_img_dir, voc_ins_seg)
    sou_img, sou_seg, voc_id = get_random_voc_img(voc_img_dir, voc_seg_dir, plot_true=False) 
    # sou_img, sou_seg = voc_img, voc_seg
    log("Get VOC image/segmentation, shape: " + str(sou_seg.shape))

    
    # ====================================================================================================
    # 2. Select Instance + Region
    log("Filter VOC instance")
    ins_seg = filter_color(sou_seg, filter_color=(255,255,255))
    ins_seg_gray = cv2.cvtColor(ins_seg, cv2.COLOR_BGR2GRAY)

    ins_img = sou_img.copy() # => No need to copy in production, just for testing
    ins_img[~ins_seg_gray > 0] = [0,0,0]

    # - (optional) resize VOC instance
    sou_bbox = regionprops(ins_seg_gray)[0].bbox

    # Randomly define resize_factor
    h, w = sou_bbox[2]-sou_bbox[0], sou_bbox[3]-sou_bbox[1]
    log("bbox: " + str(sou_bbox) + " h: " + str(h) + " w: " + str(w))
    fy = tar_shape[0] / h
    fx = tar_shape[1] / w

    resize_factor = min(fy, fx)*cover_factor # => Cover 80% of the target image
    log("Resize factor: " + str(fy) + " " + str(fx) + " " + str(resize_factor))
    dsize=(int(w*resize_factor), int(h*resize_factor))

    log("Crop/Resize VOC instance" + str(ins_seg.shape))
    
    ins_seg = crop_resize(ins_seg, sou_bbox, dsize)
    ins_seg_gray = crop_resize(ins_seg_gray, sou_bbox, dsize)
    ins_img = crop_resize(ins_img, sou_bbox, dsize)

    log("Crop/Resize VOC instance" + str(ins_seg.shape))
    # - (optional) random rotation (not implemented)

    # ====================================================================================================

    # 3. Insert VOC instance into Fashion image
    # - Select random position
    x = np.random.randint(0, tar_shape[1]-ins_seg.shape[1])
    y = np.random.randint(0, tar_shape[0]-ins_seg.shape[0])
    
    # - Create mask and inverse mask for the instance
    tar_mask = np.zeros_like(tar_img)
    h,w = ins_seg.shape[:2]
    tar_mask[y:y+h, x:x+w] = ins_seg
    inverse_mask = cv2.bitwise_not(tar_mask)

    # ====================================================================================================

    tar_img_prep = cv2.bitwise_and(tar_img, inverse_mask)
    final_seg = cv2.bitwise_and(tar_seg, inverse_mask)
    final_img = tar_img.copy()
    final_img[y:y+h, x:x+w] = cv2.add(tar_img_prep[y:y+h, x:x+w], ins_img)

    
    return tar_img, tar_seg, sou_img, sou_seg, final_img, final_seg
    # images = zip([tar_img, tar_seg, 
    #         sou_img, sou_seg, 
    #         ins_seg, ins_seg_gray, ins_img, tar_mask, inverse_mask, tar_img_prep, final_img, final_seg],
    #         ["Fashion IMG", "Fashion Segmentation", "VOC IMG", "VOC segmentation", 
    #         "Instance seg.", "Instance seg.", "Instance IMG", "tar_mask", "inverse_mask", "tar_img_prep", "Final IMG", "Final seg."])


    # log("Plot")
    # imshow_many(images, cols=3, rows=4, axis_off=False)
    # log("Finish")       
    
    