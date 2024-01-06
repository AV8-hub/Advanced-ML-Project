peut-être que j'évalue pas sur le modèle totalement entraîné vu les valeurs de loss
peut-être que y a un problème dans les données d'entrée vu à quoi elles ressemblent (regarder toutes les fonctions de dataloader, qund je resize, est-ce que je resize aussi le mask? oui je pense)
## Sports Ball Image Segmentation

3A ENSAE 2023/2024 : Advanced ML Project

BORDES Martin and ROSUNEE Avichaï

This project focuses on leveraging neural network models for accurate image segmentation of sports balls using the COCO dataset. We explore state-of-the-art architectures, including U-Net, Mask R-CNN, and DeepLab, adapting them to diverse sports scenarios. The goal is to enhance sports analytics by providing precise segmentation in dynamic settings.

If launching on SSP Cloud : put this on the terminal -> 
sudo apt-get update
sudo apt-get install libgl1-mesa-glx

sudo apt-get update
sudo apt-get install libglib2.0-0

results time : CustomUnet 30 epochs->6h 50 epochs->7h30


### Getting the data

The first step is to download the annotations here : http://images.cocodataset.org/annotations/annotations_trainval2017.zip 

Then, you should create a folder named 'COCOdataset2017' where you put two sub-folders : "images" and "annotations". In "annotations", you should put the two annotation files you just downloaded (corresponding to train and val). In "images", you should create two sub-folders "train" and "val".

Finally, you can launch this on your terminal to retrieve the desired images using the annotations : 
```bash
python get_data.py
```


### Training your model

In order to train a model, you can use the train.py file. You should write the exact model you want to train by giving the name of the class (do not forget to put "models." before) after --model. You should also specify the number of epochs after --n-epochs. the default values are models.UNetMobileNetV2fixed and 3. 
```bash
python train.py --model models.UNetMobileNetV2fixed --n-epochs 3
```
The corresponding trained model will be stored in the folder "saved models".


### Evaluating your model
After training it, a model can be evaluated using the evaluate.py file. As before, you should specify the type of model and the number of epochs. Be careful to call for a model that has already been trained.
```bash
python evaluate.py --model models.UNetMobileNetV2fixed --n-epochs 3
```
