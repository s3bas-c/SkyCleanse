import numpy as np
import cv2
import os
from glob import glob

def clamp01(x):
    return np.clip(x, 0, 1)

# ----------------------------
# 1. Noise
# ----------------------------

def add_noise(img, severity):
    severity = clamp01(severity)
    
    noise = np.random.normal(0, severity * 0.1, img.shape)
    out = img + noise
    
    return np.clip(out, 0, 1)

# ----------------------------
# 2. Blur
# ----------------------------

def add_blur(img, severity):
    severity = clamp01(severity)
    
    k = int(1 + severity * 15)
    if k % 2 == 0:
        k += 1
    
    return cv2.GaussianBlur(img, (k, k), 0)

# ----------------------------
# 3. Star trailing
# ----------------------------

def add_star_trails(img, severity):
    severity = clamp01(severity)

    length = int(severity * 32)

    if length < 2:
        return img

    if length % 2 == 0:
        length += 1

    kernel = np.zeros((length, length), dtype=np.float32)

    center = length // 2

    angle = np.random.uniform(0, 2 * np.pi)

    dx = np.cos(angle)
    dy = np.sin(angle)

    for i in range(length):

        offset = i - center

        x = int(round(center + dx * offset))
        y = int(round(center + dy * offset))

        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0

    # normalize kernel
    kernel /= kernel.sum() + 1e-8

    return cv2.filter2D(img, -1, kernel)

# ----------------------------
# 7. Haze
# ----------------------------

def add_haze(img, severity):
    severity = clamp01(severity)

    haze = np.ones_like(img) * severity * 0.4
    return np.clip(img + haze, 0, 1)

# ----------------------------
# MAIN PIPELINE
# ----------------------------

def degrade_image(img,
                  noise=0,
                  blur=0,
                  trails=0,
                  haze=0):

    img = img.astype(np.float32)

    img = add_blur(img, blur)
    img = add_star_trails(img, trails)
    img = add_haze(img, haze)
    img = add_noise(img, noise)

    return img

# ----------------------------
# QUALITY SCORE
# ----------------------------

def quality_score(noise, blur, trails, haze):

    severity = np.mean([
        noise, blur, trails, haze
    ])

    score = (1 - severity) * 10

    return float(np.clip(score, 0, 10))

def image_to_patches(img, size=128, stride=128):
    patches = []
    h, w = img.shape

    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            patches.append(img[y:y+size, x:x+size])

    return patches


def save_patch_image(img, path):

    out = (img * 255).clip(0, 255).astype(np.uint8)

    cv2.imwrite(path, out)

# ----------------------------
# DATASET BUILDER
# ----------------------------

def build_dataset(image_folder, patch_size=128, save_preview=True):
    X = []
    Y = []

    images = glob(os.path.join(image_folder, "*"))

    os.makedirs("preview_patches", exist_ok=True)

    preview_count = 0

    for path in images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        patches = image_to_patches(img, patch_size)

        for p in patches:

            noise = np.random.rand()
            blur = np.random.rand()
            trails = np.random.rand()
            haze = np.random.rand()

            degraded = degrade_image(p, noise, blur, trails, haze)

            total_score = noise * 0.25 + blur * 0.2 + trails * 0.4 + haze * 0.15

            X.append(degraded)
            Y.append(total_score)

            if save_preview and preview_count < 10:

                save_patch_image(
                    p,
                    f"preview_patches/{preview_count}_clean.png"
                )

                save_patch_image(
                    degraded,
                    f"preview_patches/{preview_count}_degraded.png"
                )

                preview_count += 1

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y

X, Y = build_dataset("Train_images", patch_size=128, save_preview=True)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

np.save("astro_X.npy", X)
np.save("astro_Y.npy", Y)
