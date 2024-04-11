import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from PIL import Image
import mmcv
from matplotlib import patches
import cv2

data_dir = "../../data/"

fashion_dir = f"{data_dir}/fashionpedia/"
annotations_dir = f"{fashion_dir}/Annotations/"


def generate_palette(num_classes, seed=42):
    """
    Generate a palette for MMSegmentation given the number of classes.

    Args:
        num_classes (int): Number of classes.

    Returns:
        numpy.ndarray: Palette array of shape (num_classes, 3).
    """
    np.random.seed(seed)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    # Generate RGB colors
    for i in range(num_classes):
        if i == 0:
            # The first color is reserved for background
            palette[i] = [0, 0, 0]
            continue
        else:
            r = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            b = np.random.randint(0, 256)
            palette[i] = [r, g, b]

    return palette


def plot_seg_image(img_id, palette, classes, fashion_dir, annotations_dir, dir="train/"):
    """Plot the original image and its segmentation mask."""
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Read and plot original image 
    img = mmcv.imread(fashion_dir + f'{dir}{img_id}.jpg')
    ax[0].imshow(mmcv.bgr2rgb(img))

    # Plot segmentation mask
    img = Image.open(annotations_dir + f'{dir}{img_id}_seg.png')

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Extract unique attributes from the image
    unique_attributes = np.unique(img_array)

    # Create a custom colormap from the filtered palette
    cmap = plt.matplotlib.colors.ListedColormap(palette[unique_attributes]/255.)

    # boundaries = np.arange(min(unique_attributes)-0.5, max(unique_attributes) + 1.5, 1)
    boundaries = np.append(unique_attributes, unique_attributes[-1]+1)
    norm = plt.matplotlib.colors.BoundaryNorm(boundaries, cmap.N)
    
    im = ax[1].imshow(img_array, cmap=cmap, norm=norm)

    # Create patches for the legend
    patches = [mpatches.Patch(color=np.array(palette[attr])/255., label=classes[attr]) for attr in unique_attributes if attr in classes]
    # Create a legend in the second subplot
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Create a custom colorbar as the legend
    cbar = plt.colorbar(im, ticks=unique_attributes, orientation='vertical')
    cbar.ax.set_yticklabels([classes[attr] for attr in unique_attributes])
    cbar.set_label('Class', rotation=270, labelpad=15)
    # Axis off
    for ax_i in ax:
        ax_i.axis('off')

    fig.tight_layout()


def imshow(img, title="", ax=None):
    """Display an image with a title."""
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.imshow(img)
    ax.set_title(title)


def imshow_many(imgs, axis_off=True, cols=1, rows=1):
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*3))
    
    for ax, img in zip(axs.flatten(), imgs):
        imshow(*img, ax=ax) if isinstance(img, tuple) else imshow(img, ax=ax)
        ax.axis('off') if axis_off else None
    
    fig.tight_layout()

def mask_resize(img, instance_pixels, resize_factor=1, instance_color=None):
    """Convert all non-instance pixels to black and resize the image"""
    
    new_img = img.copy()
    new_img[~instance_pixels] = [0, 0, 0]
    
    if instance_color is not None:
        new_img[instance_pixels] = instance_color

    # Resize propotionally
    return cv2.resize(new_img, (0,0), fx=resize_factor, fy=resize_factor)


def choice_color(img_seg):
    """Random choice of a color in the segmentation image"""

    CONTOUR = [192, 224, 224]
    BACKGROUND = [0, 0, 0]
    
    # Get unique colors in the segmentation image and filter out background and contour
    img_colors = np.unique(img_seg.reshape(-1, 3), axis=0)
    filter_colors = [not np.any(np.all(color==[BACKGROUND, CONTOUR], axis=1)) for color in img_colors]
    img_colors = img_colors[filter_colors]

    # Return a random color
    return img_colors[np.random.choice(img_colors.shape[0])]


def filter_color(img, color=None, background=[0, 0, 0], filter_color=None):
    """Filter a color in the image and background the rest of the image, if color is not given, a random color is chosen."""
    
    # Given or random color
    if color is None:
        color = choice_color(img)
    
    # Create a mask where all pixels equal to the color
    mask = np.all(img == color, axis=-1)
    
    img = img.copy()
    
    # Remove the other colors
    img[~mask] = background
    
    # Change the color of the mask
    if filter_color is not None:
        img[mask] = filter_color
    
    return img


def plot_voc_seg(img_id, voc_img_dir, voc_seg_dir, bboxes=None, title=None):
    """Plot a PASCAL VOC image with its segmentation and bounding boxes.
    
    Args:
    - img_id: str, image id
    - voc_img_dir: str, path to the folder containing the images
    - voc_seg_dir: str, path to the folder containing the segmentation masks
    - bboxes: list of lists, bounding boxes coordinates
    """
    
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Read image and segmentation
    img = cv2.imread(voc_img_dir + img_id + '.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_seg = cv2.imread(voc_seg_dir + img_id + ".png" )

    ax[0].imshow(img)
    ax[1].imshow(img_seg)

    # Draw bounding box: create a Rectangle patch for each bbox
    if bboxes is not None:
        for bbox in bboxes:
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0.5, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
    
    if title is not None:
        fig.suptitle(title, y=0.85)
    
    return img, img_seg