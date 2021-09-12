import sys
import os
import torch
from torch.utils import data
from torchvision import transforms
from DatasetPascalVoc import PascalVocDatasetClass
from yolov1 import YoloV1Loss, YoloV1
from visualizer import Visualizer

def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)
# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def train_loop(dataloader, model, loss_fn, optimizer, visualizer):
    model.train()
    total_loss = 0
    size = len(dataloader.dataset)
    for batch_idx, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item[0]
        if batch_idx % 5 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f" train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            visualizer.plot_train_val(loss_train=total_loss / (batch_idx + 1))

# def test_loop(dataloader, model, loss_fn, metric_to_calculate, epoch):
#     model.eval()
#     num_batches = len(dataloader)
#     test_loss = 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X.to(device))
#             test_loss += loss_fn(pred, y.to(device)).item()
#             metric_to_calculate(pred.argmax(1), y.to(device))
#
#     mtc = metric_to_calculate.compute()
#     test_loss /= num_batches
#     print(f"Test Error: \n Accuracy: {(100*mtc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#     pierogi.plot_loss(epoch+1, 0, test_loss, 'validation')
# path to your own data and coco file
train_data_dir = 'data/pascal_voc/train'
train_coco = './data/pascal_voc/train/_annotations.coco.json'
val_data_dir = 'data/paascal_voc/test'
val_coco = './data/pascal_voc/val/_annotations.coco.json'
# select device (whether GPU or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
torch.manual_seed(666)

criterion = YoloV1Loss(7, 2, 5, 0.5)
model = YoloV1().to(device)
learning_rate = 1e-3
batch_size = 32
num_epochs = 20
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# create own Dataset
train_dataset = PascalVocDatasetClass(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )
train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)

# create own Dataset
val_dataset = PascalVocDatasetClass(root=val_data_dir,
                          annotation=val_coco,
                          transforms=get_transform()
                          )

val_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=collate_fn)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

vis = Visualizer(env='zhav1k_yolo1')

for epoch in range(1):
    train_loop(train_loader, model, criterion, optimizer, vis)

# DataLoader is iterable over Dataset
# for imgs, annotations in data_loader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)















