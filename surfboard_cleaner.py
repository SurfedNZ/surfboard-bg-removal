import os
import cv2
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model.u2net import U2NET  # Adjust if you are using U2NETP or other variant

# Optional CRF post-processing
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    dcrf = None


def load_model(model_path):
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def remove_background(image_path, model):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        d1, *_ = model(input_tensor)
    mask = d1[0][0].numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask_resized = cv2.resize(mask, image.size)
    return image, mask_resized


def apply_crf(image_np, mask_np):
    if dcrf is None:
        raise ImportError("Install pydensecrf to use CRF refinement.")
    h, w = image_np.shape[:2]
    mask_softmax = np.zeros((2, h, w), dtype=np.float32)
    mask_softmax[0] = 1.0 - mask_np
    mask_softmax[1] = mask_np
    unary = unary_from_softmax(mask_softmax)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image_np, compat=10)
    Q = d.inference(5)
    refined_mask = np.argmax(Q, axis=0).reshape((h, w))
    return refined_mask.astype(np.uint8)


def create_alpha_image(image, mask, use_crf=False):
    image_np = np.array(image)
    if use_crf:
        mask = apply_crf(image_np, mask)
    else:
        mask = (mask > 0.5).astype(np.uint8)
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack((image_np, alpha))
    return Image.fromarray(rgba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='saved_models/u2net.pth', help='Path to U2Net model')
    parser.add_argument('--output', required=True, help='Path to save final cleaned image')
    parser.add_argument('--use_crf', action='store_true', help='Apply CRF refinement')
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model)

    print("Removing background...")
    image, mask = remove_background(args.image, model)

    print("Refining and saving result...")
    result = create_alpha_image(image, mask, args.use_crf)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.save(args.output)

    print(f"✅ Cleaned image saved at: {args.output}")


if __name__ == '__main__':
    main()
