from pycocotools.coco import COCO
import requests

def get_images(folder='./COCOdataset2017', dataType='train', annpath = '{}/annotations/instances_{}2017.json'):

    # instantiate COCO specifying the annotations json path
    annFile=annpath.format(folder,dataType)
    coco=COCO(annFile)
    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=['sports ball'])
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open("./COCOdataset2017/images/train/" + im['file_name'], 'wb') as handler:
            handler.write(img_data)


