# Sports Ball Image Segmentation

3A ENSAE 2023/2024 : Advanced ML Project

BORDES Martin and ROSUNEE Avichaï

This project focuses on leveraging neural network models for accurate image segmentation of sports balls and trains using the COCO dataset but can be adapted to of the 91 classes present in the dataset. We compare a simple U-Net to a U-Net using transfer learning and the pretrained model MobileNet-V2.

If launching on SSP Cloud : put this on the terminal -> 
cd Advanced-ML-Project

pip install -r requirements.txt

sudo apt-get update
sudo apt-get install libgl1-mesa-glx

sudo apt-get update
sudo apt-get install libglib2.0-0

python get_data.py

### Installing the requirements

```bash
pip install -r requirements.txt
```

### Getting the data

You can get images from teh COCO dataset by using the get_data script. You can choose what class you want to import by specifying the argument --object. The default value is "train" but you can choose between all the 91 classes available.
```bash
python get_data.py --object "train"
```
This will create a COCOdataset2017 folder containing the images divided between the training set and the validation set, as well as the annotations (useful to construct the masks later).
WARNING : this step takes more than 30 minutes (on the Onyxia SSPCloud Datalab)


## How to recreate our results 

Recreating our results is possible by doing all over again from the training with the following steps. If you do not have hours in front of you, you can simply evaluate a model without training it by using our already trained models (skip the training step) (you should still use a GPU).

### Training your model

In order to train a model, you can use the train.py file. This file will first put our data in a good format, in particular by constructing the mask, and then train a model on it. You should write the exact model you want to train by giving the name of the class after --model. You should also specify the number of epochs, whether to add the augmentation strategy or not and on which objet to train the model. The default values are specified here:
```bash
python train.py --model CustomUnet --n-epochs 15 --augment False --object "train"
```
The corresponding trained model will be stored in the "saved models" folder.


### Evaluating your model

After training it, a model can be evaluated using the evaluate.py file. You should specify the same arguments as before
```bash
python evaluate.py --model CustomUnet --n-epochs 15 --augment False --object "train"
```
This will create a .csv file containing the loss, the accuracy as well as the IoU score, stored in the "results" folder.

## How the code was created

The get_data.py and a large part of the dataloader.py file were largely inspired by this github : https://github.com/virafpatrawala/COCO-Semantic-Segmentation. It allowed us to retrieve the images from the COCO dataset efficiently, using the pycocotools library. It is not the only library designed to import from the COCO dataset but it was the most simple to understand. To resume, this github guided us to import the data and create the masks. Then, we converted these data into the good format (the shape suitable to Pytorch and dataloaders).

Then, we were also inspired by this other github : https://github.com/hayashimasa/UNet-PyTorch. There are no parts of our code really copied from here but it guided us on the imbriquement of each step and on the format of some parts in the train and the evaluate script. We adapted it to our own project. For example, it made us discover the argparse library, really useful when working with python files.
