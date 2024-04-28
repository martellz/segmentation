import random
import torchvision.transforms as tfs
from torchvision.transforms import functional as F
from torchvision.transforms import v2


class RandomRotation(object):
  def __init__(self, degrees):
    self.degrees = degrees

  def __call__(self, sample):
    image, label = sample
    angle = random.uniform(-self.degrees, self.degrees)
    image = v2.functional.rotate(image, angle)
    label = v2.functional.rotate(label, angle)
    return (image, label)

class ColorJitter(v2.ColorJitter):
  def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
    super().__init__(brightness, contrast, saturation, hue)

  def __call__(self, sample):
    image, label = sample
    image = super().__call__(image)
    return (image, label)

class RandomHorizontalFlip(v2.RandomHorizontalFlip):
  def __init__(self, p=0.5):
    super().__init__(p)

  def __call__(self, sample):
    image, label = sample
    if random.random() < self.p:
      image = v2.functional.hflip(image)
      label = v2.functional.hflip(label)
    return (image, label)

class RandomVerticalFlip(v2.RandomVerticalFlip):
  def __init__(self, p=0.5):
    super().__init__(p)

  def __call__(self, sample):
    image, label = sample
    if random.random() < self.p:
      image = v2.functional.vflip(image)
      label = v2.functional.vflip(label)
    return (image, label)

class RandomCropMaxSquare(object):
  def __call__(self, sample):
    image, label = sample
    min_edge = min(image.shape[1], image.shape[2])
    crop_params = tfs.RandomCrop.get_params(image, (min_edge, min_edge))
    image = F.crop(image, *crop_params)
    label = F.crop(label, *crop_params)

    return image, label

class Resize(object):
  def __init__(self, size):
    self.size = size

  def __call__(self, sample):
    image, label = sample
    image = F.resize(image, self.size)
    label = F.resize(label, self.size)
    return image, label
