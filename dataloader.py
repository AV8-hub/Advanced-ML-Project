from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


def filterDataset(folder, classes, annpath, mode):
    """
    Filter the COCO dataset based on specified classes and mode (e.g., 'train' or 'val').

    Parameters:
    - folder (str): The root directory of the COCO dataset.
    - classes (list): A list of category names to filter images.
    - annpath (str): The format of the annotations json path.
    - mode (str): The data split mode, e.g., 'train' or 'val'.

    Returns:
    - unique_images (list): List of unique image data dictionaries.
    - coco (COCO): COCO API object.
    """
    # Initialize COCO API for instance annotations
    annFile = annpath.format(folder, mode)
    coco = COCO(annFile)
    images = []

    # Get all images containing given categories
    for className in classes:
        catIds = coco.getCatIds(catNms=className)
        imgIds = coco.getImgIds(catIds=catIds)
        images += coco.loadImgs(imgIds)

    # Remove repeated images
    unique_images = list({image['id']: image for image in images}.values())
    random.shuffle(unique_images)

    return unique_images, coco

def getClassName(classID, cats):
    """
    Retrieve the class name associated with a given class ID from a list of categories.

    Parameters:
    - classID (int): The ID of the class to retrieve the name for.
    - cats (list): List of category dictionaries, each containing an 'id' and 'name'.

    Returns:
    - str or None: The name of the class if found, otherwise None.
    """
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    """
    Read and normalize an image from a specified folder.

    Parameters:
    - imageObj (dict): Image data dictionary, containing 'file_name'.
    - img_folder (str): Folder path where the image is located.
    - input_image_size (tuple): Target size for the image, e.g., (height, width).

    Returns:
    - numpy.ndarray: Normalized and resized image.
    """
    # Read and normalize an image
    train_img = io.imread(os.path.join(img_folder, imageObj['file_name'])) / 255.0

    # Resize
    train_img = cv2.resize(train_img, input_image_size)

    if len(train_img.shape) == 3 and train_img.shape[2] == 3:  # If it is an RGB 3-channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img

    
def getMask(imageObj, classes, coco, catIds, input_image_size):
    """
    Generate and return segmentation mask for a given image.

    Parameters:
    - imageObj (dict): Image data dictionary, containing 'id'.
    - classes (list): List of class names.
    - coco (COCO): COCO API object.
    - catIds (list): List of category IDs.
    - input_image_size (tuple): Target size for the mask, e.g., (height, width).

    Returns:
    - numpy.ndarray: Segmentation mask corresponding to the image.
    """
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)

    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add an extra dimension for parity with train_img size [X * X * 1]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask
 


def getTensors(images, classes, coco, folder, mode, input_image_size):
    """
    Generate torch tensors for images and corresponding masks.

    Parameters:
    - images (list): List of unique image data dictionaries.
    - classes (list): List of class names.
    - coco (COCO): COCO API object.
    - folder (str): The root directory of the COCO dataset.
    - mode (str): The data split mode, e.g., 'train' or 'val'.
    - input_image_size (tuple): Target size for the tensors, e.g., (height, width).

    Returns:
    - torch.Tensor: Tensor of images.
    - torch.Tensor: Tensor of masks.
    """
    img_folder = os.path.join(folder, 'images', mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    X = []
    y = []

    for i in range(dataset_size):
        imageObj = images[i]

        ### Retrieve Image ###
        img = getImage(imageObj, img_folder, input_image_size)
        img = np.resize(img, (3, input_image_size[0], input_image_size[1]))
        X.append(img)

        ### Create Mask ###
        mask = getMask(imageObj, classes, coco, catIds, input_image_size)
        mask = np.resize(mask, (1, input_image_size[0], input_image_size[1]))
        y.append(mask)

    X = torch.Tensor(np.array(X))
    y = torch.Tensor(np.array(y))

    return X, y


def AugmentData(X, y, p=0.3):
    """
    Apply data augmentation to a set of images and masks.

    Parameters:
    - X (torch.Tensor): Tensor of images.
    - y (torch.Tensor): Tensor of masks.
    - p (float): Probability of applying each augmentation. Default is 0.5.

    Returns:
    - torch.Tensor: Augmented tensor of images.
    - torch.Tensor: Augmented tensor of masks.
    """
    n = len(X)

    for i in range(n):
        image = X[i]
        mask = y[i]

        for k in range(3):
            if random.random() < p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < p:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() < p:
                image = v2.ColorJitter(brightness=random.random())(image)

            if random.random() < p:
                noise = torch.randn(image.size()) * 0.3 + 0.5
                image = image + noise

            X = torch.cat((X, image.unsqueeze(0)), 0)
            y = torch.cat((y, mask.unsqueeze(0)), 0)

    return X, y


def getDataloader(mode, folder='./COCOdataset2017', classes=['sports ball'],
                  annpath='{}/annotations/instances_{}2017.json', input_image_size=(224, 224), batch_size=4):
    """
    Create a DataLoader for a specified mode ('train' or 'val').

    Parameters:
    - mode (str): The data split mode, e.g., 'train' or 'val'.
    - folder (str): The root directory of the COCO dataset. Default is './COCOdataset2017'.
    - classes (list): List of class names. Default is ['sports ball'].
    - annpath (str): The format of the annotations json path. Default is '{}/annotations/instances_{}2017.json'.
    - input_image_size (tuple): Target size for the images, e.g., (height, width). Default is (224, 224).
    - batch_size (int): Batch size for the DataLoader. Default is 64.

    Returns:
    - DataLoader: PyTorch DataLoader for the specified mode.
    """
    images, coco = filterDataset(folder, classes, annpath, mode)
    X, y = getTensors(images, classes, coco, folder, mode, input_image_size)

    if mode == 'train':
        X, y = AugmentData(X, y)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader


