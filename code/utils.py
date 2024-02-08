import torch
import torchvision
from .dataset import Dataset
from torch.utils.data import DataLoader
import pathlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    val_ds = Dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda", epoch=1):
    num_correct=0
    num_pixels=0
    dice_score =0

    writer = SummaryWriter("Accuracy_plot")
    step = 0
    sum_acc = 0

    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / (
                (preds+y).sum() + 1e-8
            )

            acc = 100 * (num_correct/num_pixels)
            step += 1
            sum_acc += acc
        
        writer.add_scalar("Training accuracy", sum_acc/step, global_step=epoch)
            
    print(
        f"Got {num_correct}/{num_pixels} with acc {100 * (num_correct/num_pixels):.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
    
def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    pathlib.Path(folder).mkdir(exist_ok=True)
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x=x.to(device=device)
        with torch.no_grad():
            #preds=torch.sigmoid(model(x))
            #preds=(preds>0.5).float()
            preds=torch.argmax(model(x), 1).float()

            preds = preds[0].cpu().detach().numpy()

            plt.figure()
            plt.imshow(preds, cmap="tab20", vmin=0, vmax=12)
            plt.colorbar()
            plt.show()
            plt.savefig(f"{folder}/pred_{idx}.png")
            plt.close()
            plt.cla()
            plt.clf()
    

        
        target = torch.argmax(y, 1).float()

        target = target[0].cpu().detach().numpy()

        plt.figure()
        plt.imshow(target, cmap="tab20", vmin=0, vmax=12)
        plt.colorbar()
        plt.show()
        plt.savefig(f"{folder}/target_{idx}.png")
        plt.close()
        plt.cla()
        plt.clf()


        
    model.train()