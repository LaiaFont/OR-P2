import xml.etree.ElementTree as ET
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import json
import pandas as pd
import shutil


# Read JSON annotations
def read_annotations(anno_file):
    """Read annotations from json file and return a DataFrame with annotations and images info."""

    with open(anno_file) as json_data:
        data = json.load(json_data)
        categories = pd.DataFrame(data['categories'])
        df_img = pd.DataFrame(data['images'])
        annotations = pd.DataFrame(data['annotations'])
    
    df_annot = pd.merge(annotations, df_img, left_on='image_id', right_on='id', how='outer')
    df_annot.drop(columns=['id_x', 'id_y', 'license', 'time_captured', 'isstatic', 'original_url', 'iscrowd', 'kaggle_id'], inplace=True)
    
    # Compute img area and area ratio
    df_annot["img_area"] = df_annot["height"] * df_annot["width"]
    df_annot['area_ratio'] = df_annot['area'] / df_annot['img_area']
    
    return df_annot, categories, df_img, annotations


# Read and format data (extracted from given base notebook)
def read_voc_xml(xml_file: str, voc_classes: dict):
    """Function to read XML file and return the list of objects and their bounding boxes."""

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_objects = []
    for boxes in root.iter('object'):

        classname = boxes.find("name").text
        list_with_all_objects.append(voc_classes[classname])

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = round(float(boxes.find("bndbox/ymin").text))
        xmin = round(float(boxes.find("bndbox/xmin").text))
        ymax = round(float(boxes.find("bndbox/ymax").text))
        xmax = round(float(boxes.find("bndbox/xmax").text))
        
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_objects, list_with_all_boxes


def get_voc_img(voc_id, voc_img_dir, voc_seg_dir, plot_true=False):
    """Select given ID image and its segmentation from VOC dataset"""
    
    # print(f"VOC ID: {voc_id}")
    # print(f"Image: {voc_img_dir + voc_id + '.jpg'}")
    # print(f"Segmentation: {voc_seg_dir + voc_id + '.png'}")
    
    # print(len(os.listdir(voc_img_dir)))
    
    # Read images
    img_file = voc_id + ".jpg"
    seg_file = voc_id + ".png"
    img = cv2.imread(voc_img_dir + img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_seg = cv2.imread(voc_seg_dir + seg_file)
    
    # Plot images
    if plot_true:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for ax, im, file in zip(axs, [img, img_seg], [img_file, seg_file]):
            ax.imshow(im)
            ax.set_title(file)
            ax.axis('off')
        
        fig.suptitle(img_file)
        fig.tight_layout()
        
    return img, img_seg, voc_id


def get_random_voc_img(voc_img_dir, voc_seg_dir, plot_true=False):
    """Randomly select an image and its segmentation from VOC dataset"""
    
    # Choice a segmentation file and its corresponding image
    seg_file = random.choice(os.listdir(voc_seg_dir))
    
    # Delete the '.png' extension
    voc_id = seg_file[:-4]
    
    return get_voc_img(voc_id, voc_img_dir, voc_seg_dir, plot_true)
    

def get_fashion_img_seg(img_id, img_dir, seg_dir, plot=False):
    """
    segment.png = '9f98c5425e9f04d3c7def0bb2af2771d_seg.png'
    image.jpg = '9f98c5425e9f04d3c7def0bb2af2771d.jpg'
    """
    
    # Read image and segment
    img = cv2.imread(os.path.join(img_dir, f"{img_id}.jpg"))
    img_seg = cv2.imread(os.path.join(seg_dir, f"{img_id}_seg.png"))
    
    # Parse to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Plot images
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for ax, im in zip(axs, [img, img_seg]):
            ax.imshow(im)
            # ax.set_title(file)
            ax.axis('off')
        
        fig.suptitle(img_id)
        fig.tight_layout()
        
    return img, img_seg
    
    
    return image, segment
    

# Save data to json file
def save_json(filepath, data):
    """Save data to json file"""
    with open(filepath, 'w') as f:
        json.dump(data, f)


# Get the directory size
def get_dir_size(dir_path):
    """Get directory size"""
    return subprocess.check_output(['du','-sh', dir_path]).split()[0].decode('utf-8')


# Reduce data to N images or list of categories, create new json file and folder
def reduce_data(json_file, img_dir, new_img_dir, new_json_file, n_choice=1000, categories=None):
    """Reduce data to N images and save to new json file and folder"""

    # Load json file
    # data.keys() => dict_keys(['annotations', 'images', 'info', 'licenses', 'categories', 'attributes'])
    print(f"Original JSON file size: {os.path.getsize(json_file)/(1024**2):.2f}MB ({json_file})")
    with open(json_file) as json_data:
        data = json.load(json_data)   
    
    
    # Get image/annotations data and remove unnecessary columns and parse to DataFrame
    df_annots = pd.DataFrame(data['annotations'])
    df_images = pd.DataFrame(data['images'])[['id', 'width', 'height', 'file_name']]
    
    if categories is not None:
        print(f"Filtering data to {len(categories)} categories")
        
        # Create new dataframes with selected categories
        df_annots_new = df_annots[df_annots.category_id.isin(categories)]
        img_choice = df_annots_new.image_id.unique()
        n_choice = len(img_choice)
        
        df_images_new = df_images[df_images.id.isin(img_choice)]
           
    else:
    
        # Get unique image and category IDs
        unique_imgs = np.unique(df_images.id)
        # unique_cats = np.unique(df_annots.category_id)

        # Randomly select N images
        img_choice = np.random.choice(unique_imgs, n_choice, replace=False)
          
        # Filter data
        df_annots_new = df_annots[df_annots.image_id.isin(img_choice)]
        df_images_new = df_images[df_images.id.isin(img_choice)]

    
    print(f"Reducing data to {n_choice} images...")
    print(f"Original Image-dir size: {get_dir_size(img_dir)} ({img_dir})")
    print(f"Original num. images: {df_images.shape[0]}")
    print(f"Original Annotations: {df_annots.shape[0]}")
    
    print(f"New num. images: {df_images_new.shape[0]}")
    print(f"New Annotations: {df_annots_new.shape[0]}")
    

    # Create new data dictionary
    new_data = {}
    new_data["annotations"] = df_annots_new.to_dict(orient='records')
    new_data["images"] = df_images_new.to_dict(orient='records')
    
    if categories is not None:
        new_data["categories"] = [cat for cat in data["categories"] if np.isin(cat["id"], categories).sum()]
    else:
        new_data["categories"] = data["categories"]

    # Copy images to new folder
    os.makedirs(new_img_dir, exist_ok=True)
    print("Copying images to new folder...")
    for i, row in df_images_new.iterrows():
        
        # Get image name and old/new paths
        img_name = row.file_name
        img_path = os.path.join(img_dir, img_name)
        new_img_path = os.path.join(new_img_dir, img_name)
        
        # Copy image to new folder
        shutil.copyfile(img_path, new_img_path)

    print(f"New Image-dir size: {get_dir_size(new_img_dir)} ({new_img_dir})")
    
    # Save new data to json file
    save_json(new_json_file, new_data)
    print(f"New JSON file size: {os.path.getsize(new_json_file)/(1024**2):.2f}MB ({new_json_file})")


# Create a masks (PNG) for each category
def generate_masked_imgs(fp, save_dir, palette, convert_3D=True, n_images=None):
    """Generate segmentation masks for each category in the dataset."""
    
    
    im_ids = fp.getImgIds()
    if n_images is not None:
        im_ids = im_ids[:n_images]

    counter = 0
    print('Number of images: {}'.format(len(im_ids)))
    print(f'Saving files to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)

    # Get annotations IDs for each image 
    d3 = True
    for im_id in im_ids:
        ann_ids = np.array(fp.getAnnIds(imgIds=im_id))
                
        # Create a list of category IDs for each annotation
        cat_ids = []
        for i in ann_ids:
            cat_ids.append(fp.anns[i]['category_id']+1)

        # Sort the annotations by category ID
        ann_ids = ann_ids[np.argsort(cat_ids)]

        # Create a mask for each category
        mask = None
        for i in ann_ids:

            # Get the segmentation mask (1 or 0) for the annotation
            seg_mask = fp.annToMask(fp.anns[i])
            color = (fp.anns[i]['category_id']+1)
            
            # Apply color to the mask
            m = seg_mask * color
            
            # Cumulative mask
            mm = (m>0).astype(np.uint8)
            if mask is None:
                mask = m
            else:
                mask = mask*(1-mm) + m

        # Convert the mask to 3D if needed
        if convert_3D:
            mask3d = np.zeros(shape=(*mask.shape, 3), dtype=np.uint8)
            for cat in np.unique(mask):
                mask3d[mask == cat] = palette[cat]
            mask = mask3d
                
        # Save the mask to PNG file in the 'ann_dir' directory
        filename = os.path.join(save_dir, fp.imgs[im_id]['file_name'].split('.')[0]+'_seg.png')
        if not cv2.imwrite(filename, mask):
            print(f'Failed to save {filename}, stopping.')
            break
        
        counter += 1
        if counter % 1000 == 0:
            print(f'{counter} images processed.')

        
    print(f'Finished! {counter} images processed.')
  