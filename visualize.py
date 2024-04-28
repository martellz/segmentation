import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from nets.deeplabv3_plus import DeepLab


def visualize(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=16, pretrained=False)
  model.load_state_dict(torch.load(args.ckpt))
  model.to(device)
  model = model.eval()

  original_image = cv2.imread(args.image).astype(np.float32) / 255.0
  h, w = original_image.shape[:2]
  image = cv2.resize(original_image, (512, 512))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  image = torch.from_numpy(image).to(device)
  image = image.permute(2, 0, 1).unsqueeze(0)

  with torch.no_grad():
    output = model(image)
    output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
    output = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    weights = np.where(output == 1, 1, 0.1)
    mask = np.where(output == 1, 255, 0).astype(np.uint8)
    plt.imshow(weights[..., np.newaxis] * original_image)
    plt.show()

    # draw countours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    cv2.drawContours(original_image, contours, -1, (0, 1.0, 0), 2)
    plt.imshow(original_image)
    plt.show()

    original_image = (original_image * 255).astype(np.uint8)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('contour.jpg', original_image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--image', type=str, default='data')
  parser.add_argument('--ckpt', type=str, default=None, help='resume from checkpoint')
  parser.add_argument('--num_class', type=int, default=2)
  args = parser.parse_args()
  visualize(args)