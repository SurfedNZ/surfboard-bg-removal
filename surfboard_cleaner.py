import os
import cv2
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from model.u2net import U2NETP  # Adjust if you are using U2NETP or other variant

# Optional CRF post-processing
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    dcrf = None


def load_model(model_path):
    model = U2NETP(3, 1)
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

def process_image(image_path, output_path, model, use_crf):
    image, mask = remove_background(image_path, model)
    result = create_alpha_image(image, mask, use_crf)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"✅ Processed {os.path.basename(image_path)} → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to input image (single image mode)')
    parser.add_argument('--output', help='Path to save cleaned image (single image mode)')
    parser.add_argument('--input_dir', help='Directory with input images for batch processing')
    parser.add_argument('--output_dir', help='Directory to save cleaned images (batch mode)')
    parser.add_argument('--model', default='saved_models/u2net.pth', help='Path to U2Net model')
    parser.add_argument('--use_crf', action='store_true', help='Apply CRF refinement')
    
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        parser.error("You must provide either --image for single image or --input_dir for batch processing.")

    if args.image and not args.output:
        parser.error("--output is required when using --image.")

    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir.")

    print("Loading model...")
    model = load_model(args.model)

    if args.image:
        print("Processing single image...")
        process_image(args.image, args.output, model, args.use_crf)

    if args.input_dir:
        print(f"Processing batch images from {args.input_dir}...")
        supported_exts = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(supported_exts)]
        os.makedirs(args.output_dir, exist_ok=True)

        for fname in image_files:
            input_path = os.path.join(args.input_dir, fname)
            output_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + '_clean.png')
            process_image(input_path, output_path, model, args.use_crf)



if __name__ == '__main__':
    main()
