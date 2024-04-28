import argparse
import torch

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pathlib import Path
from nets.deeplabv3_plus import DeepLab
from utils.dataset import WaterDataset
from utils.transforms import Resize
from train import weighted_crossentropyloss, val_epoch

def test(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print('Loading dataset. ')
  dataroot = Path(args.root)
  transforms = v2.Compose([
    Resize([512, 512]),
  ])

  val_set = WaterDataset(dataroot / 'val', transforms=transforms, device='cpu', cache_data=True)
  val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
  model = DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=16, pretrained=False)
  model.load_state_dict(torch.load(args.ckpt))
  model.to(device)
  model = model.eval()

  val_results_path = Path('val_results')
  val_results_path.mkdir(exist_ok=True)
  val_loss, val_pixel_acc, val_mean_iou = val_epoch(val_loader, val_results_path, model, weighted_crossentropyloss, args.num_class, device)
  print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_pixel_acc:.4f}, Val mIOU: {val_mean_iou:.4f}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', type=str, default='data')
  parser.add_argument('--ckpt', type=str, default=None, help='resume from checkpoint')
  parser.add_argument('--num_class', type=int, default=2)
  args = parser.parse_args()
  test(args)