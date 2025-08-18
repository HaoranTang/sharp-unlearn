import numpy as np
from io import BytesIO
from PIL import Image as PILImage
import cv2
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import torchvision.transforms as T

def motion_blur_kernel(kernel_size, angle):
    """Create a normalized linear motion‐blur kernel."""
    # Start with a horizontal line
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[(kernel_size-1)//2, :] = 1.0
    kernel /= kernel_size
    # Rotate it to the desired angle
    M = cv2.getRotationMatrix2D(
        (kernel_size/2 - 0.5, kernel_size/2 - 0.5),
        angle, 1.0
    )
    return cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img_cropped = img[top:top + ch, top:top + ch]
    zoomed = scizoom(img_cropped, (zoom_factor, zoom_factor, 1), order=1)
    trim = (zoomed.shape[0] - h) // 2
    return zoomed[trim:trim + h, trim:trim + h]

def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]
    x = np.array(x).astype(np.float32) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]
    arr = np.array(x).astype(np.float32) / 255.0
    # initial blur
    blurred = gaussian(arr, sigma=c[0], multichannel=True)
    x_blur = np.uint8(blurred * 255)
    # local pixel shuffling
    for _ in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dy, dx = np.random.randint(-c[1], c[1], size=2)
                h2, w2 = h + dy, w + dx
                x_blur[h, w], x_blur[h2, w2] = x_blur[h2, w2], x_blur[h, w]
    # final blur
    final = gaussian(x_blur.astype(np.float32)/255.0, sigma=c[0], multichannel=True)
    return np.clip(final, 0, 1) * 255

def snow(x, severity=1):
    # c = (mean, std, zoom, thresh, radius, sigma, blend_alpha)
    c = [
        (0.1,0.2,1,0.6,  8, 3, 0.95),
        (0.1,0.2,1,0.5, 10, 4, 0.90),
        (0.15,0.3,1.75,0.55,10,4,0.90),
        (0.25,0.3,2.25,0.6,12,6,0.85),
        (0.3,0.3,1.25,0.65,14,12,0.80),
    ][severity - 1]

    arr = np.array(x).astype(np.float32) / 255.0
    # make monochrome snow layer
    mono = np.random.normal(size=arr.shape[:2], loc=c[0], scale=c[1])
    mono = clipped_zoom(mono[..., np.newaxis], c[2]).squeeze()
    mono[mono < c[3]] = 0

    # === REPLACED: WandImage motion_blur with OpenCV kernel ===
    # convert to 0–255 grayscale for blurring
    mono_u8 = (mono * 255).astype(np.uint8)
    # choose a random angle for motion direction
    angle = np.random.uniform(-135, -45)
    # build and apply the motion‐blur kernel
    kernel = motion_blur_kernel(c[4], angle)
    blurred = cv2.filter2D(mono_u8, -1, kernel).astype(np.float32) / 255.0
    snow_arr = blurred[..., np.newaxis]
    # === end replacement ===

    # blend with color image
    base = c[6] * arr + (1 - c[6]) * np.maximum(
        arr,
        cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).reshape(32,32,1)*1.5 + 0.5
    )
    combined = base + snow_arr + np.rot90(snow_arr, k=2)
    return np.clip(combined, 0, 1) * 255

# ------------------------------------------------------------
# Transform wrappers
# ------------------------------------------------------------
class GaussianNoiseTransform:
    def __init__(self, severity=3):
        self.severity = severity
    def __call__(self, img: PILImage) -> PILImage:
        out = gaussian_noise(img, severity=self.severity)
        return PILImage.fromarray(out.astype(np.uint8))

class GlassBlurTransform:
    def __init__(self, severity=3):
        self.severity = severity
    def __call__(self, img: PILImage) -> PILImage:
        out = glass_blur(img, severity=self.severity)
        return PILImage.fromarray(out.astype(np.uint8))

class SnowTransform:
    def __init__(self, severity=3):
        self.severity = severity
    def __call__(self, img: PILImage) -> PILImage:
        out = snow(img, severity=self.severity)
        return PILImage.fromarray(out.astype(np.uint8))