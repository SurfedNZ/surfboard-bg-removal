import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2

from model.u2net import U2NET  # Update if using u2netp or other variant

# Optional CRF (requires: pip install pydensecrf)
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


def refine_mask(mask, threshold=0.9):
    mask = np.clip(mask, 0, 1).astype(np.float32)
    _, binary = cv2.threshold(mask, threshold, 1.0, cv2.THRESH_BINARY)
    binary = (binary * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def apply_crf(img, mask):
    if not CRF_AVAILABLE:
        print("CRF module not installed. Skipping CRF step.")
        return mask

    h, w = mask.shape
    d = dcrf.DenseCRF2D(w, h, 2)
    prob = np.stack([1 - mask, mask], axis=0)
    unary = unary_from_softmax(prob)
    d.setUnaryEnergy(unary)

    feats = create_pairwise_gaussian(sdims=(3, 3), shape=(h, w))
    d.addPairwiseEnergy(feats, compat=3)

    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=img, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    Q = d.inference(5)
    result = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)
    return result * 255


def preprocess_image(pil_image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(pil_image).unsqueeze(0)
    return image_tensor


def load_model(model_path, use_gpu=True):
    net = U2NET(3, 1)
    if use_gpu and torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net


def remove_background(image_path, model_path, output_path, use_crf=False):
    image = Image.open(image_path).convert("RGB")
    original_np = np.array(image)

    image_tensor = preprocess_image(image)
    image_tensor = Variable(image_tensor)

    net = load_model(model_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    d1, *_ = net(image_tensor)
    pred = d1[0][0].cpu().data.numpy()
    pred = cv2.resize(pred, (original_np.shape[1], original_np.shape[0]))
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    refined = refine_mask(pred, threshold=0.9)

    if use_crf:
        refined = apply_crf(original_np, refined.astype(np.float32) / 255.0)

    alpha = refined.astype(np.float32) / 255.0
    image_np = original_np.astype(np.float32) / 255.0

    rgba = np.dstack((image_np, alpha))
    rgba = (rgba * 255).astype(np.uint8)
    out_image = Image.fromarray(rgba)
    out_image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refine surfboard background removal")
    parser.add_argument('--image', required=True, help='Path to input surfboard image')
    parser.add_argument('--model', required=True, help='Path to trained U²-Net .pth file')
    parser.add_argument('--output', default='output.png', help='Path to save transparent image')
    parser.add_argument('--use_crf', action='store_true', help='Use CRF refinement (slower)')
    args = parser.parse_args()

    remove_background(args.image, args.model, args.output, args.use_crf)
