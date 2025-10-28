import os
from PIL import Image
import numpy as np
import cv2

def save_preprocessed_image(image, isic_id, output_dir):
    output_path = os.path.join(output_dir, f"{isic_id}.jpg")
    image.save(output_path, format='JPEG')

def preprocess_image(image, target_size=224):
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    preprocessed = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    preprocessed[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return Image.fromarray(preprocessed)

def process_image(args):
    image_path, id, output_dir = args
    img = Image.open(image_path).convert('RGB')
    img_pre = preprocess_image(img)
    save_preprocessed_image(img_pre, id, output_dir)
