import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pathlib import Path
from tqdm import tqdm
from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.dataset import WaterDataset
from utils.transforms import ColorJitter, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomCropMaxSquare, Resize


def mean_iou(pred, label, num_classes):
  iou = torch.zeros((pred.shape[0]))
  for idx in range(num_classes):
    out1 = (pred == idx)
    out2 = (label == idx)

    intersect = torch.sum(out1 & out2, dim=(1, 2)).type(torch.FloatTensor)
    union = torch.sum(out1 | out2, dim=(1, 2)).type(torch.FloatTensor)
    iou += torch.div(intersect, union + 1e-16)
  m_iou = torch.mean(iou) / num_classes
  return m_iou

def pixel_acc(pred, label):
  acc = torch.eq(pred, label).type(torch.FloatTensor).mean()
  return acc

def weighted_crossentropyloss(output, target, device, n_classes):
  """ Weighted Cross Entropy Loss"""
  # 以前的代码暂时有点问题，先不用
  # n_pixel = target.numel()  #number of pixels
  # _, counts = torch.unique(target, return_counts=True)
  # cls_weight = torch.div(n_pixel, n_classes * counts.type(torch.FloatTensor)).to(device)
  # loss = F.cross_entropy(output, target, weight=cls_weight)

  loss = F.cross_entropy(output, target)

  return loss

def train_epoch(loader, model, criterion, optimizer, n_classes, device):
  loss_sum = 0.0
  train_pixel_acc = []
  train_mean_iou = []

  model.to(device)
  model.train()

  for i, sample_batched in enumerate(loader):
    image, label = sample_batched
    image = image.to(device)
    label = label.to(device)

    target = label[:, 0, :, :]
    # target = torch.argmax(label, dim=1)
    optimizer.zero_grad()

    outputs = model(image)
    out_cat = torch.argmax(outputs, dim=1)

    loss = criterion(outputs, target, device=device, n_classes=n_classes)

    loss.backward()
    optimizer.step()

    loss_sum += loss.item()
    train_pixel_acc.append(pixel_acc(out_cat, target))
    train_mean_iou.append(mean_iou(out_cat, target, n_classes))

  return [loss_sum, np.mean(train_pixel_acc), np.mean(train_mean_iou)]

def val_epoch(loader, val_results_path: Path, model, criterion, n_classes, device):
  loss_sum = 0.0
  val_pixel_acc = []
  val_mean_iou = []

  model.to(device)
  model.eval()
  with torch.no_grad():
    for i, sample_batched in enumerate(loader):
      image, label = sample_batched
      image = image.to(device)
      label = label.to(device)
      target = label[:, 0, :, :]
      # target = torch.argmax(label, dim=1)

      outputs = model(image)
      out_cat = torch.argmax(outputs, dim=1)

      loss = criterion(outputs, target, device=device, n_classes=n_classes)

      loss_sum += loss.item()
      val_pixel_acc.append(pixel_acc(out_cat, target))
      val_mean_iou.append(mean_iou(out_cat, target, n_classes))

      out_cat = out_cat.cpu().detach().numpy()
      for j, cat in enumerate(out_cat):
        cat = np.squeeze(cat)
        cat = np.where(cat == 1, 255, 0).astype(np.uint8)
        cv2.imwrite(str(val_results_path / '{}_{}.png'.format(i, j)), cat)

    return [loss_sum, np.mean(val_pixel_acc), np.mean(val_mean_iou)]

def train(args):
  seed = 233
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  nbs             = 16
  batch_size      = 8
  max_epoch       = 1000
  lr_limit_max    = 5e-4
  lr_limit_min    = 3e-4

  Init_lr_fit     = min(max(batch_size / nbs * lr_limit_max, lr_limit_min), lr_limit_max)
  Min_lr_fit      = min(max(batch_size / nbs * lr_limit_max * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

  lr_scheduler_func = get_lr_scheduler('cos', Init_lr_fit, Min_lr_fit, max_epoch)

  print('Loading dataset. ')
  dataroot = Path(args.root)
  transforms = v2.Compose([
    # RandomCropMaxSquare(),
    Resize([512, 512]),
    ColorJitter(brightness=[0.75, 1.25], contrast=[0.75, 1.25], saturation=[0.75, 1.25], hue=[-0.25, 0.25]),
    RandomRotation(25),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
  ])
  train_set = WaterDataset(dataroot / 'train', transforms=transforms, device='cpu', cache_data=True)
  # val_set = WaterDataset(dataroot / 'val', transforms=None, device='cpu', cache_data=True)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
  # val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

  print('Creating model. ')
  model = DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=16, pretrained=True)
  # model = SimpleNet(num_classes=2)
  weights_init(model)

  if args.ckpt:
    model.load_state_dict(torch.load(args.ckpt))

  model = model.to(device)
  model = model.train()

  optimizer = Adam(model.parameters(), lr=Init_lr_fit, betas=(0.9, 0.999), weight_decay=1e-4)

  with tqdm(range(1, max_epoch + 1)) as pbar:
    for epoch in pbar:
      set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

      loss_sum, train_pixel_acc, train_mean_iou = train_epoch(train_loader, model, weighted_crossentropyloss, optimizer, args.num_class, device)

      print({'Loss': loss_sum, 'Acc': train_pixel_acc, 'mIOU': train_mean_iou})

      # if epoch % 50 == 0:
      #   torch.cuda.empty_cache()
      #   val_results_path = Path('val_results')
      #   val_results_path.mkdir(exist_ok=True)
      #   val_loss, val_pixel_acc, val_mean_iou = val_epoch(val_loader, val_results_path, model, weighted_crossentropyloss, args.num_class, device)
      #   torch.cuda.empty_cache()
      #   print(f'Epoch: {epoch}, Train Loss: {loss_sum:.4f}, Train Acc: {train_pixel_acc:.4f}, Train mIOU: {train_mean_iou:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_pixel_acc:.4f}, Val mIOU: {val_mean_iou:.4f}')

      if epoch % 100 == 0 and epoch != 0:
        ckpt_save_dir = Path('checkpoints')
        ckpt_save_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_save_dir / f'deeplab_v3_3_epoch_{epoch}.pth')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='data')
  parser.add_argument('--ckpt', type=str, default=None, help='resume from checkpoint')
  parser.add_argument('--num_class', type=int, default=2)
  args = parser.parse_args()
  train(args)