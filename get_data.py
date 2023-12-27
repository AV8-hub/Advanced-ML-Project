from pycocotools.coco import COCO
import requests

def get_images(dataType, folder='./COCOdataset2017', annpath = '{}/annotations/instances_{}2017.json', classes=['sports ball']):

    # instantiate COCO specifying the annotations json path
    annFile=annpath.format(folder,dataType)
    coco=COCO(annFile)
    # Specify a list of category names of interest
    catIds = coco.getCatIds(catNms=classes)
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    # Save the images into a local folder
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open(f"./COCOdataset2017/images/{dataType}/" + im['file_name'], 'wb') as handler:
            handler.write(img_data)

if __name__ == '__main__':
    get_images(dataType = 'train')
    get_images(dataType = 'val')

