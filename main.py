# Intialization
import os
import sys
import cv2
import time
import random
import numpy as np
import nibabel as nib
import torch
from sklearn.preprocessing import binarize
from torch import nn
import torch.optim as optim
sys.path.append('/Users/sahilnalawade/Desktop/Sahil/Projects/python/organ_segmentation/liver_segmentation/')
import torch.nn.functional as F
from tiramisu_3d import *
from torchmetrics.functional.classification import dice_score, dice
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from functools import partial

# Functions

# Read Flist -- To Read Flist

def read_flist(flist_filepath):
    '''
    Function to Read a flist file
    flist_file = flist file to be read
    :return: flist file (each line for for each filepath)
    '''
    with open(flist_filepath, 'r') as file_in:
        file = [line.rstrip('\n') for line in file_in]
    return file


# Normalize Image

def normalize_image(img):
    '''
    Function to normalize the imaging data along each axis
    :param img:
    :return:
    '''

    mean = img.mean(axis=(0,1,2))
    stddev = img.std(axis=(0,1,2))
    img_norm = img - mean
    img_norm = img_norm / stddev

    return img_norm


# Read Image

def read_image(image_filepath, normalize=True, reference=False, shape_adjust=(256, 256, 155)):
    '''
    Read Nifti Image
    :return:
    '''

    img_ref = nib.load(image_filepath)
    img_data = img_ref.get_fdata()

    if normalize:
        img_data = normalize_image(img=img_data)
        img_data = np.array(img_data)

    # To adjust the shape of the image --> Blank Image * ones * Image.min
    # img_ones = np.ones(shape_adjust) * img_data.min()
    # img_ones[:240, :240, :] = img_data

    if reference:
        return img_data, img_ref
    else:
        return img_data

def tme():
    '''
    Function for getting the current time
    :return:
    current time
    '''
    return time.time()


def check_overlap_patchsize_per_axis(isz1, psz1):
    '''
    Fucntion to Check the overlap of each axis
    :param isz1: size of image along '1' axis
    :param psz1: size of patch along '1' axis
    :return:
    nwsz1 : modified size of the patch along '1' axis
    '''

    dv1 = ((isz1 - psz1) % psz1)
    nwsz1 = (isz1 + psz1 - dv1) if dv1 != 0 else isz1
    return nwsz1


def create_new_image(inp_img, patch_size):
    '''
    Function To check the Image dimension for overlapping Patches and if require create a new image with new dimensions
    :param inp_img: 3D array [H,W,D]
    :param patch_size: 4D array [N,H,W,D]
    :return:
    new_image: 3D array with [new_H,new_W,new_D]
    '''

    isz1, isz2, isz3 = inp_img.shape
    psz1, psz2, psz3 = patch_size

    # Create New dimension -- If the overlapping patches exceed the dimension
    nwsz1 = check_overlap_patchsize_per_axis(isz1, psz1)
    nwsz2 = check_overlap_patchsize_per_axis(isz2, psz2)
    nwsz3 = check_overlap_patchsize_per_axis(isz3, psz3)
    new_image_dimension = [nwsz1, nwsz2, nwsz3]

    # Create new Image -- Copy the old image into it
    new_image = np.ones(new_image_dimension) * inp_img.min()
    new_image[:isz1, :isz2, :isz3] = inp_img

    return new_image


def get_overlap_strides(patch_size, overlap_percent):
    '''
    Function to get overlap strides using overlap percentage
    :param patch_size: list for patch dimension [patch_h, patch_w, patch_d]
    :param overlap_percent: Overlap Percentage between 0 and 0.9 [0.5, 0.6, 0.]
    :return:
    overlap_size: Overlap size along each axis [overlap_h, overlap_w, overlap_d]
    '''
    psz1, psz2, psz3 = patch_size
    ov1, ov2, ov3 = overlap_percent
    osz1, osz2, osz3 = int(ov1 * psz1), int(ov2 * psz2), int(ov3 * psz3)
    overlap_size = osz1, osz2, osz3

    return overlap_size


def get_list_of_indices(new_img, patch_size, overlap_size):
    '''
    Function to get the list of indices along each axis
    :param new_img: 3d array [H,W,D]
    :param patch_size: list for patch dimension [patch_h, patch_w, patch_d]
    :param overlap_size: list for overlap strides [stride_h, stride_w, stride_d]
    :return:
    indices: list of indices for each axis [[0, 10 ...], [0, 50 ..], [0, 100 ...]]
    '''
    nwsz1, nwsz2, nwsz3 = new_img.shape
    psz1, psz2, psz3 = patch_size
    osz1, osz2, osz3 = overlap_size

    x_idx = np.arange(0, nwsz1-psz1+1, psz1-osz1)
    y_idx = np.arange(0, nwsz2-psz2+1, psz2-osz2)
    z_idx = np.arange(0, nwsz3-psz3+1, psz3-osz3)
    indices = [x_idx, y_idx, z_idx]

    return indices


def append_pacthes(inp_img, indices, patch_size, binary_map=False):
    '''
    Function to append patches
    :param inp_img: image 3D Array [H,W,D]
    :param indices: list of Indices across axis
    :param patch_size: patch size across axis [patch_height, patch_width, patch_dept]
    :param binary_map: Bool Variable (True: Generate the Map, False: Ignore)
    :return:
    patch_array_3d : appended patch array [N,H,W,D]
    bin_map: 3D array of dimension [H,W,D]
    '''

    psz1, psz2, psz3 = patch_size
    patch_array_3d = []
    bin_map = np.zeros(inp_img.shape)

    for idx1 in indices[0]:
        for idx2 in indices[1]:
            for idx3 in indices[2]:
                patch = inp_img[idx1:idx1+psz1, idx2:idx2+psz2, idx3:idx3+psz3]
                patch_array_3d.append(patch)

                # Binary Map for recovering the patch
                if binary_map:
                    bin_map[idx1:idx1+psz1, idx2:idx2+psz2, idx3:idx3+psz3] += 1

    return patch_array_3d, bin_map


def get_3d_image_from_indices(inp_img_ptch, new_image, indices, patch_size):
    '''
    Function to get a 3D Image from patches
    :param inp_img_ptch: 4D patch array [N, patch_h, patch_w, patch_d]
    :param new_image: 3D array [H, W, D] -- array with all zeros
    :param indices: list of Indices across axis
    :param patch_size: patch size across axis [patch_height, patch_width, patch_dept]
    :return:
    mod_new_image: final 3D array [H, W, D] with data extracted from all patches (inp_img_ptch)
    '''
    psz1, psz2, psz3 = patch_size
    x_idx, y_idx, z_idx = indices
    c1 = 0
    mod_new_image = new_image.copy()
    for idx1 in x_idx:
        for idx2 in y_idx:
            for idx3 in z_idx:
                mod_new_image[idx1:idx1 + psz1, idx2:idx2 + psz2, idx3:idx3 + psz3] += inp_img_ptch[c1]
                c1 += 1

    return mod_new_image


def create_3d_patches_from_image(inp_img, patch_size, overlap_percent):
    '''
    Function To extract Patches from 3D Images
    :param inp_img: image 3D Array [H, W, D]
    :param patch_size: patch size across axis [patch_height, patch_width, patch_dept]
    :param overlap_percent: percentage value for patch overlap ranging between 0 and 0.9 [0.5, 0.6, 0.]
    :return:
    patch_array_3d: 4D patch array [N, patch_h, patch_w, patch_d]
    bin_map: 3D array [H1, W1, D1] giving the counts for all the voxels
    '''
    # Get all the input variables
    overlap_size = get_overlap_strides(patch_size, overlap_percent)

    # Create new image and get new image dimension
    new_image = create_new_image(inp_img, patch_size)

    # Get the patch Indices across axis
    indices = get_list_of_indices(new_image, patch_size, overlap_size)

    # Function Call :: To get pacthes from the 3D Image
    patches_for_3d_image_array, bin_map = append_pacthes(new_image, indices, patch_size, binary_map=True)

    return patches_for_3d_image_array, bin_map


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceMetric(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceMetric, self).__init__()

    def forward(self, inputs, targets, argmax=False, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        if argmax:
            inputs = torch.argmax(inputs, dim=1)
            targets = torch.argmax(targets, dim=1)
            # print(inputs.shape, targets.shape)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


def sel_optimizer(optim_name):

    optimizer = None
    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    return optimizer


def sel_criterion(criterion_name):

    criterion = None
    if criterion_name == 'dice_bce':
        criterion = DiceBCELoss()

    if criterion_name == 'dice':
        criterion = DiceLoss()

    return criterion


def sel_metrics(metrics_name):
    '''

    :param metrics_name:
    :return:
    '''
    metrics = None
    if metrics_name == 'dice':
        metrics = partial(DiceMetric())
    if metrics_name == 'jaccard': # This is also known as IOU
        metrics = partial(JaccardIndex)

    return metrics


def sel_model(model_name):
    '''
    Select model for the task (segmentation) based on the name
    :param model_name: String mentioning the name of the model
    :return:
    model: A deep learning architecture for the task (Segmentation)
    '''
    model = None
    if model_name == 'densenet':
        model = FCDenseNetcustom(n_classes=1)

    return model


def prepare_input(input_tuple):
    '''
    Function to get the input data for training or validation
    :param input_tuple: Tuple of (Input, Target)
    :return:
    input: Input Data mapped to a device is returned
    target: Target Data mapped to a device is returned
    '''
    input, target = input_tuple
    input = input.to(device)
    target = target.to(device)
    return input, target


class Seg_Data(Dataset):
    def __init__(self, image, label, transform=None, target_transform=None):

        self.image = image
        self.label = label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image, label = self.image[idx], self.label[idx]
        image = torch.from_numpy(image[None, :])
        label = torch.from_numpy(label[None, :])
        # label = nn.functional.one_hot(torch.tensor(np.array(label, dtype='int64')), num_classes=2)
        # label = np.squeeze(np.swapaxes(label, 0, -1))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train(epoch, model, train_data_loader, criterion, metrics):

    # Train the model
    model.train()
    for batch_idx, input_tuple in enumerate(train_data_loader):

        # clear the gradients
        optimizer.zero_grad()

        input_tensor, target = prepare_input(input_tuple)
        input_tensor.requires_grad = True

        # compute the model output
        output = model(input_tensor.float())
        loss = criterion(output, target)
        score = metrics(output, target)

        loss.backward()
        optimizer.step()

        print('[{}:: Batch: {}, iter: {}, loss: {:.3f}, score: {:.3f}]'.format('Train', batch_idx, epoch * len_epoch + batch_idx, loss.item(), score))
        if (batch_idx + 1) % terminal_show_freq == 0:
            partial_epoch = epoch + batch_idx / len_epoch - 1



def validate_epoch(epoch, model, valid_data_loader, criterion, metrics):

    # Evaluate the model
    val_score = []
    model.eval()
    for batch_idx, input_tuple in enumerate(valid_data_loader):
        with torch.no_grad():

            input_tensor, target = prepare_input(input_tuple)
            input_tensor.requires_grad = False

            output = model(input_tensor.float())
            loss = criterion(output, target)
            score = metrics(output, target)

            print('[{}:: Batch: {}, iter: {}, loss: {:.3f}, score: {:.3f}]'.format('Valid', batch_idx, epoch * len_epoch + batch_idx, loss.item(), score))
            val_score.append(score)
    return val_score

config = {
    'file_path': '/Users/sahilnalawade/Desktop/Sahil/Projects/Dataset/medical_image_decathalon/liver_lesion_segmentation/flist/input.flist',
    'model': 'Unet',
    'encoder': 'densenet169',
    'in_channels': 1,
    'out_classes': 2,
    'epoch': 40,
    'terminal_show_freq': 1,
    'optimizer': 'adam',
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'criterion': 'dice',
    'metrics': 'dice',
    'model_name': 'densenet',
    'validation_iteration': 2,
    'model_path': '/Users/sahilnalawade/Desktop/Sahil/Projects/python/organ_segmentation/liver_segmentation/model/dense_model_v1.pth',

}

# Get Data
files = read_flist(config['file_path'])
files = files[:2]
print('Total Files: {}'.format(len(files)))
img_lst = []
lbl_lst = []

for idx in range(len(files)):

    image_path = files[idx]
    file_parts = files[idx].split('/')
    folder_path = ('/').join(file_parts[:-1])
    folder_name = file_parts[-2]
    image_last_name = file_parts[-1]
    label_last_name = 'segmentation_' + image_last_name.split('_')[-1]
    label_path = os.path.join(folder_path, label_last_name)

    img = read_image(image_filepath=image_path, normalize=True, reference=False)
    lbl = read_image(image_filepath=label_path, normalize=False, reference=False)
    lbl = np.round(lbl)
    lbl = np.where(lbl > 1.0, 1.0, 0.0)

    ptch_3d_img, _ = create_3d_patches_from_image(inp_img=img, patch_size=(32,32,32), overlap_percent=(0, 0, 0))
    ptch_3d_lbl, _ = create_3d_patches_from_image(inp_img=lbl, patch_size=(32,32,32), overlap_percent=(0, 0, 0))

    [sl, _, _, _] = np.nonzero(ptch_3d_lbl)
    sl = np.unique(sl)
    # sl = random.choices(sl, k=10)
    ptch_3d_img = [ptch_3d_img[idx] for idx in sl]
    ptch_3d_lbl = [ptch_3d_lbl[idx] for idx in sl]

    print('#{} -- Reading Subject : {}, Slices : {}, Len of Patch : {}, label : {} ...'.format(idx+1, folder_name, len(sl), len(ptch_3d_lbl), np.unique(ptch_3d_lbl)))

    img_lst.extend(ptch_3d_img)
    lbl_lst.extend(ptch_3d_lbl)

print('Image : {}, Label : {}'.format(len(img_lst), len(lbl_lst)))
print('Labels : {}'.format(np.unique(lbl_lst)))

# Model -- Intialize
model = sel_model(config['model_name'])

# Parameters -- Training
optimizer = sel_optimizer(config['optimizer'])
epoch = config['epoch']
terminal_show_freq = config['terminal_show_freq']
device = config['device']
criterion = sel_criterion(config['criterion'])  # Loss
metrics = sel_metrics(config['metrics'])
val_iter = config['validation_iteration']
print("Device: ", device)

# arr_ = ptch_3d_img
# shape_ = (len(arr_), 2,) + arr_[0].shape
# labels = torch.randint(0, 2, shape_)

# Data:: getting the dataset and dataloaders
training_data = Seg_Data(image=img_lst, label=lbl_lst)
validation_data = Seg_Data(image=img_lst, label=lbl_lst)

train_data_loader = DataLoader(training_data, batch_size=3, shuffle=False)
valid_data_loader = DataLoader(validation_data, batch_size=3, shuffle=False)
len_epoch = len(train_data_loader)

def main(epoch, model, train_data_loader, valid_data_loader, criterion, metrics, val_iter):

    temp = 100
    for idx in range(config['epoch']):
        print('\nStart Training Epoch #{} ....\n'.format(idx+1))
        train(epoch, model, train_data_loader, criterion, metrics)

        if (idx + 1) % val_iter == 0:
            print('\nStart Validation Epoch #{} ....\n'.format(idx+1))
            val_score = validate_epoch(epoch, model, valid_data_loader, criterion, metrics)

            print('Val Score :{:.3f}'.format(np.mean(val_score)))
            if np.mean(val_score) < temp:
                torch.save(model.state_dict(), config['model_path'])
                temp = np.mean(val_score)

if __name__ == "__main__":
    main(epoch, model, train_data_loader, valid_data_loader, criterion, metrics, val_iter)


print('---- COMPLETED ----')
