# depthest

============================

## Contents
0. [Requirements](#requirements)
0. [Dataset](#dataset)
0. [Training](#training)
0. [Resume from a particular checkpoint](#resume-from-a-particular-checkpoint)
0. [Evaluation](#evaluation)
0. [Inference](#inference)

## Requirements
- Install CUDA 11.7.0
- Programming language: Python 3.7.9
- Install all the requirements 
```bash
pip install -r requirements.txt
```

## Dataset
Download the preprocessed KITTI Odometry dataset in HDF5 format and place it under a `kitti` folder inside the repo directory. The KITTI Odometry dataset requires 82G of storage space.
```bash
mkdir kitti; cd kitti
wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
cd ..
```

## Training 
arch - architecture 
epoch - number of epochs
b - batch size 
data - dataset name
t - train 

To train the MobileNetV3SkipAddL-NNConv5R architecture:
```bash
python main.py -t train --arch MobileNetV3SkipAddL_NNConv5R --epoch 10 -b 8 --data kitti
```
To train the MobileNetV3SkipAddL-NNConv5S architecture:
```bash
python main.py -t train --arch MobileNetV3SkipAddL_NNConv5S --epoch 10 -b 8 --data kitti
```
To train the MobileNetV3L-NNConv5GU architecture:
```bash
python main.py -t train --arch MobileNetV3L_NNConv5GU --epoch 10 -b 8 --data kitti
```
To train the MobileNetV3SkipAddS-NNConv5R architecture:
```bash
python main.py -t train --arch MobileNetV3SkipAddS_NNConv5R --epoch 10 -b 8 --data kitti
```
To train the MobileNetV3S-NNConv5GU architecture:
```bash
python main.py -t train --arch MobileNetV3S_NNConv5GU --epoch 10 -b 8 --data kitti
```

## Resume from a particular checkpoint
resume - resume training from a particular checkpoint

To resume from a particular checkpoint use the following command:
```bash
python main.py --resume results/MobileNetV3SkipAddS-NNConv5R/checkpoint_10.pth.tar
```

## Evaluation
evaluation - start the evaluation process 

To evaluate the model use the following command:      
```bash
python main.py --evaluate results/MobileNetV3SkipAddS-NNConv5R/MobileNetV3SkipAddS-NNConv5R.pth.tar
```

## Inference
model - the name of the trained model

To run the real-time inference use the following command:
```bash
python inference.py 
```
By default the MobileNetV3SkipAddS-NNConv5R is inferred. To run any other model:
```bash
python inference.py --model “name of the needed trained model”
```
