# Liver Segmentation

## Task
This project is about organ segmentation focused on liver. 
Segmenting liver will be help in extracting the liver component for further task.

## Data
Public Dataset: LITS 2017 data
Competition Page - https://competitions.codalab.org/competitions/17094

## Architecture

Architecture used is 3d-Dense-U-net for liver segmentation

## Training
1. The model was trtained on Tesla T-4 (16 GB)
2. The Data was split into Training - 70%, Validation - 20% and testing - 10%
3. Loss used for training the algorithm is dice loss.
4. Metrics used for training and validation is dice score.


## Testing
1. The testing is performed using a pre-trained model
2. The model is trained on LITS 2017 dataset
3. Testing can be performed with the model using a 'python test.py'

