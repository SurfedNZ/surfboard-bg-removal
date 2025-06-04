📁 README.md

# 🏄 Surfboard Background Removal & Refinement

This project provides an end-to-end pipeline to remove the background from surfboard images using a U²-Net deep learning model. It also supports optional CRF post-processing to refine the mask for cleaner, sharper results — ideal for surfboard marketplaces and listings.

---

## 🔧 Features

- ✅ Background removal using [U²-Net](https://github.com/xuebinqin/U-2-Net)
- ✨ Optional mask refinement using DenseCRF
- 🖼️ Outputs clean `.png` images with transparent background
- 🧵 One single script to run the entire process

---

## 📦 Requirements

Install the following Python packages:

```bash

pip3 install opencv-python pillow numpy pydensecrf torch torchvision

```

📁 Project Structure

```

surfboard-bg-removal-main/
├── model/
│   └── u2net.py           # U²-Net architecture
├── saved_models/
│   └── u2net.pth          # Pretrained model weights (download manually)
├── input_images/
│   └── surfboard1.jpg     # Example input
├── output_images/
│   └── surfboard1_clean.png # Clean output
├── surfboard_cleaner.py   # ✅ All-in-one script
└── README.md              # This file

```

🚀 Quick Start
Download the U²-Net pretrained weights

Download from the official repo and place it in saved_models/u2net.pth:
📥 U²-Net Weights – Google Drive

Run the full pipeline

```bash

# Single image mode
python surfboard_cleaner.py \
  --image input_images/surfboard1.jpg \
  --output output_images/surfboard1_clean.png \
  --model model/u2netp.pth

# Batch mode (processes all images in input_dir)
python surfboard_cleaner.py \
  --input_dir input_images \
  --output_dir output_images \
  --model model/u2netp.pth

# Batch mode (With CRF Refinement), ADD --use_CRF to the base of Single image mode for Single image option with CRF also
python surfboard_cleaner.py \
  --input_dir input_images \
  --output_dir output_images \
  --model model/u2netp.pth
  --use_crf


```

✅ Omit --use_crf to skip refinement.

🛠️ Options
Argument	Description	Default
--image	Path to input image	required
--model	Path to U²-Net model file	saved_models/u2net.pth
--output	Path to save final image	required
--use_crf	Apply optional CRF mask refinement	off by default

📸 Result Example
Input Image	Cleaned Output

(Update docs/ with your own before/after images for display.)

💡 Use Case
This tool is built for surfboard resellers, shapers, and marketplaces that want:

Clean listing photos

Consistent visuals across inventory

Automated photo prep with minimal editing

🧠 Acknowledgments

U²-Net by Xuebin Qin et al.

DenseCRF by Philipp Krähenbühl and Vladlen Koltun

🔒 License
MIT License – free to use and modify.




