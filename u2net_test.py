import torch
import os
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
from model.u2net import U2NETP

# Image pre-processing
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    model_path = os.path.join("model", "u2netp.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def remove_bg(input_image):
    try:
        image = input_image.convert("RGB")
    except Exception as e:
        raise ValueError("Error processing input image: ensure it's a valid image format.") from e

    net = load_model()

    # Preprocess image
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        d1, _, _, _, _, _, _ = net(img)
        pred = d1[:, 0, :, :]
        pred = pred.squeeze().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min())  # normalize

    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(image.size)

    # Ensure result is RGBA with white background
    image = image.convert("RGBA")
    image.putalpha(mask)
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    result = Image.alpha_composite(white_bg, image)

    return result

if __name__ == "__main__":
    test_image_path = "input.jpg"
    output_path = "output.png"

    try:
        image = Image.open(test_image_path)
        result = remove_bg(image)
        result.save(output_path)
        print(f"✅ Saved processed image to {output_path}")
    except FileNotFoundError:
        print(f"❌ File '{test_image_path}' not found.")
    except UnidentifiedImageError:
        print("❌ The file is not a valid image.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
