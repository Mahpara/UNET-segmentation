import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from .model import UNET
from .utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
from tensorboardX import SummaryWriter
import argparse
import splitfolders
import os
import sys
import shutil

# Argparse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str, required=True, help='defines the dataset')
parser.add_argument('-l', '--learning_rate', default=1e-4, help='learning rate')
parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
parser.add_argument('-e', '--epochs', type=int, required=True, help='number of epochs')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--width', type=int, default=300, help='image width')
parser.add_argument('-c', '--classes', type=int, default=13, help='number of classes')

args = parser.parse_args()

# Hyperparameters
LEARNING_RATE = args.learning_rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
NUM_WORKERS = 2
IMAGE_HEIGHT = args.height
IMAGE_WIDTH = args.width
PIN_MEMORY = True
LOAD_MODEL = True
DATASET = args.dataset
TRAIN_IMG_DIR = 'output/train/train_images'
TRAIN_MASK_DIR = 'output/train/train_segmentation'
VAL_IMG_DIR = 'output/val/train_images'
VAL_MASK_DIR = 'output/val/train_segmentation'
NUM_CLASSES = args.classes

# If a folder exists, delete it.
if os.path.exists("output"):
    shutil.rmtree("output")
if os.path.exists("saved_images"):
    shutil.rmtree("saved_images")
    
# Splitting folders
splitfolders.ratio(DATASET, output="output", seed=1337, ratio=(.8, 0.1,0.1)) 

writer = SummaryWriter()
step = 0

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    #global global_writer_step 
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            global step

            writer.add_scalar('Training Loss', loss, global_step = step) # changed step to epoch
            #writer.add_histogram("ups, ConvTranspose2d", model.ups[0].weight)
            writer.flush()
            step += 1
            
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def main():
    train_transform = A.Compose(
        [
            
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.crops.transforms.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, always_apply=True, p=1.0),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        
        ],
    )
    
    val_transform = A.Compose(
        [
        #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        #A.Rotate(limit=35, p=1.0),
        #A.HorizontalFlip(p=0.5),
        A.Normalize(
        mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    model = UNET(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)
        
        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE, epoch=epoch)
        #writer.add_scalar('Accuracy', acc, epoch) # changed here ====================
        #writer.flush()

        
        
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE 
        )
    
if __name__ == "__main__":
    main()