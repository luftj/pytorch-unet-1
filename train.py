from collections import defaultdict
import torch.nn.functional as F
import torch.nn as nn
from loss import dice_loss
from torch.utils.data import DataLoader, random_split
from dataset import BasicDataset
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

from pytorch_unet import UNet

dir_img = "C:/projects/phd/unet_map_seg/tiles_100k/imgs/"#'E:/data/train/AB_tiles/imgs/'
dir_mask = "C:/projects/phd/unet_map_seg/tiles_100k/masks/"#'E:/data/train/AB_tiles/masks/'
dir_checkpoint = ""#'E:/experiments/deepseg_models/checkpoints16/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

img_scale = 1.0
val_percent = 0.1
batch_size = 1
pos_weight = 60
epochs = 4
dataset = BasicDataset(dir_img, dir_mask, img_scale)
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])

dataloaders = {
    'train': DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    'val': DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
}

criterion = nn.BCEWithLogitsLoss(pos_weight=(torch.cuda.FloatTensor([pos_weight]) if torch.cuda.is_available() else torch.FloatTensor([pos_weight])) ) # class weighting loss fxn
def calc_loss(pred, target, metrics, bce_weight=0.5):
    # bce = F.binary_cross_entropy_with_logits(pred, target)

    # pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    # loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = criterion(pred, target)

    # metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for batch in dataloaders[phase]:
                inputs = batch['image']
                labels = batch['mask']
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    num_class = 1

    model = UNet(num_class).to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-5)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    print("start training")
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs)