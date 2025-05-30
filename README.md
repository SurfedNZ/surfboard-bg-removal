# Surfboard Background Removal API

A lightweight Flask API using U²-Net for background removal, optimized for surfboard product shots.

## Setup

1. Install requirements:

```
pip install -r requirements.txt
```

2. Add the pre-trained U²-Net model file (`u2netp.pth`) to the `model/` directory.

3. Run the app:

```
python app.py
```

4. POST an image to `/remove-background` to receive a transparent PNG.