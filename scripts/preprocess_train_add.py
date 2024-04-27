import argparse
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

def preprocess(input_dir, output_dir):
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  train_dir = output_dir / 'JPEGImages'
  label_dir = output_dir / 'SegmentationClass'

  output_dir.mkdir(parents=True, exist_ok=True)
  train_dir.mkdir(parents=True, exist_ok=True)
  label_dir.mkdir(parents=True, exist_ok=True)

  for img_path in tqdm(input_dir.glob('*.jpg'), desc='Processing'):
    mask_path = input_dir / (img_path.stem + '_mask.png')
    # roboflow exports masks with shape (H, W) and value 1 is water label
    # so we need to convert it to format same as provided dataset
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask = mask[..., np.newaxis].astype(np.uint8)
    mask = mask * 128
    mask = np.concatenate([np.zeros_like(mask), np.zeros_like(mask), mask], axis=-1)

    mask_path = label_dir / (img_path.stem + '.png')
    cv2.imwrite(str(mask_path), mask)

    # the image in provided dataset is .png format(although it's still 3-channel image)
    # so we convert .jpg image of roboflow to .png
    img = cv2.imread(str(img_path))
    img_path = train_dir / (img_path.stem + '.png')
    cv2.imwrite(str(img_path), img)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  args = parser.parse_args()

  preprocess(args.input_dir, args.output_dir)