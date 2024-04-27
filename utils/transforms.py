import random
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
