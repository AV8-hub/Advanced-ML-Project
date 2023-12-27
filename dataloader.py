from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import torch
from torch.utils.data import TensorDataset, DataLoader


def filterDataset(folder, classes, annpath, mode, ):    
    # initialize COCO api for instance annotations
    annFile = annpath.format(folder, mode)
    coco = COCO(annFile)
    images = []
    for className in classes:
        # get all images containing given categories
        catIds = coco.getCatIds(catNms=className)
        imgIds = coco.getImgIds(catIds=catIds)
        images += coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            
            unique_images.append(images[i])
            
    random.shuffle(unique_images)
    
    return unique_images, coco

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img
    
def getMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask  


def getTensors(images, classes, coco, folder, mode, input_image_size):
    
    img_folder = '{}/images/{}'.format(folder, mode)
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
        
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    
    return X, y

def AugmentData(X, y, p = 0.5):

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
                noise = torch.randn(image.size())*0.3 + 0.5
                image = image + noise
                image = v2.ColorJitter(brightness=random.random())(image)
            
            X = torch.cat((X,image.unsqueeze(0)),0)
            y = torch.cat((y,mask.unsqueeze(0)),0)

    return X, y

def getDataloader(mode, folder = './COCOdataset2017', classes=['sports ball'], annpath = '{}/annotations/instances_{}2017.json', input_image_size=(224,224), batch_size = 64):

    images, coco = filterDataset(folder, classes, annpath, mode)
    X, y = getTensors(images, classes, coco, folder, mode, input_image_size)
    if mode == 'train':
        X, y = AugmentData(X, y)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=8)
    return dataloader

