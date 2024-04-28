import cv2
import numpy as np
import torch
from typing import List
from pathlib import Path
from torch.utils.data.dataset import Dataset


class WaterDataset(Dataset):
  root: Path
  cache_data: bool
  images: List[str | torch.Tensor] = []
  labels: List[str | torch.Tensor] = []

  def __init__(self, root:str | Path, transforms=None, device: str | torch.device='cuda', cache_data=True):
    self.root = root if isinstance(root, Path) else Path(root)
    self.transforms = transforms    # transforms should be custom!
    self.device = device
    self.cache_data = cache_data  # as dataset is small, we can cache all data in gpu memory
    for image_path in self.root.glob('JPEGImages/*.png'):
      label_path = self.root / 'SegmentationClass' / (image_path.stem + '.png')
      if cache_data:
        image, label = self.load_images(image_path, label_path, device)
        self.images.append(image)
        self.labels.append(label)
      else:
        self.images.append(str(image_path))
        self.labels.append(str(label_path))

  def __len__(self):
    assert len(self.images) == len(self.labels)
    return len(self.images)

  def __getitem__(self, index):
    if self.cache_data:
      image, label = self.images[index], self.labels[index]
    else:
      image, label = self.load_images(self.images[index], self.labels[index], self.device)

    if self.transforms:
      image, label = self.transforms((image, label))

    return image, label

  @staticmethod
  def load_images(image_path, label_path, device):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).to(device)
    image = image.permute(2, 0, 1)

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    label = torch.from_numpy(label).to(device)
    label = label[..., 0:1]  # only one channel is used in provided dataset
    label = torch.where(label > 0, torch.tensor(1, device=device), torch.tensor(0, device=device))
    label = label.permute(2, 0, 1)
    return image, label


# test and visualize
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from torchvision.transforms import v2
  from transforms import ColorJitter, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomCropMaxSquare, Resize

  transforms = v2.Compose([
    RandomCropMaxSquare(),
    Resize([512, 512]),
    ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.5, 0.5]),
    RandomRotation(15),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
  ])

  dataset = WaterDataset('newdata/train', transforms, 'cpu', False)
  print('dataset length: {}'.format(len(dataset)))
  image, label = dataset[20]
  plt.imshow(image.squeeze().permute(1, 2, 0))
  plt.show()
  plt.imshow(label.permute(1, 2, 0))
  plt.show()
