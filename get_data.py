from pycocotools.coco import COCO
import requests
import os
from tqdm import tqdm
from zipfile import ZipFile

def get_coco_annotations(data_split='trainval', annotation_dir='COCOdataset2017'):
    """
    Download and extract COCO dataset annotations for a specified data split.

    Parameters:
    - data_split (str): The data split to download, e.g., 'trainval' (default).
    - annotation_dir (str): The directory to store the downloaded and extracted annotations.
                           Default is 'COCOdataset2017'.

    Returns:
    None
    """
    # Specify the COCO dataset annotation URL
    coco_annotation_url = f'http://images.cocodataset.org/annotations/annotations_{data_split}2017.zip'
    
    # Create the annotation directory if it does not exist
    os.makedirs(annotation_dir, exist_ok=True)

    # Download the zipped annotation file
    zip_filename = f'annotations_{data_split}2017.zip'
    zip_filepath = os.path.join(annotation_dir, zip_filename)

    print(f"Downloading {data_split} annotations...")
    response = requests.get(coco_annotation_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(zip_filepath, 'wb') as zip_file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            zip_file.write(data)

    progress_bar.close()
    print(f"Downloaded {data_split} annotations to: {zip_filepath}")

    # Extract the contents of the zip file
    with ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(annotation_dir)

    # Remove the downloaded zip file
    os.remove(zip_filepath)

    print(f"Extracted {data_split} annotations to: {annotation_dir}")


def get_images(dataType, folder='./COCOdataset2017', annpath='{}/annotations/instances_{}2017.json', classes=['sports ball']):
    """
    Download and save COCO dataset images for a specified data type and class categories.

    Parameters:
    - dataType (str): The type of data to download, e.g., 'train' or 'val'.
    - folder (str): The root directory to store the downloaded images. Default is './COCOdataset2017'.
    - annpath (str): The format of the annotations json path. Default is '{}/annotations/instances_{}2017.json'.
    - classes (list): A list of category names of interest. Default is ['sports ball'].

    Returns:
    None
    """
    # Create the images, train, and val directories
    os.makedirs(os.path.join(folder, 'images', dataType), exist_ok=True)

    # Instantiate COCO specifying the annotations json path
    annFile = annpath.format(folder, dataType)
    coco = COCO(annFile)

    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=classes)

    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder with a progress bar
    for im in tqdm(images, desc=f"Downloading {dataType} images for {', '.join(classes)}"):
        img_data = requests.get(im['coco_url']).content
        with open(os.path.join(folder, 'images', dataType, im['file_name']), 'wb') as handler:
            handler.write(img_data)


if __name__ == '__main__':
    get_coco_annotations(data_split='trainval')
    get_images(dataType = 'train')
    get_images(dataType = 'val')

