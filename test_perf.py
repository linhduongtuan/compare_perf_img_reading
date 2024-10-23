import os
import time
import cv2
import glob
from torchvision import io
from PIL import Image
import pyspng
from memory_profiler import profile


@profile
def test_torchvision_read_image(AB_path):
    start_time = time.time()
    AB = io.read_image(AB_path, mode=io.ImageReadMode.RGB)
    w = AB.shape[2]
    if w > AB.shape[1]:
        w2 = int(w / 3)
        A = AB[:, :, :w2]
        B1 = AB[:, :, w2:w2*2]
        B2 = AB[:, :, w2*2:]
    else:
        w2 = int(w / 2)
        A = AB[:, :, :w2]
        B1 = AB[:, :, w2:]
        B2 = B1
    end_time = time.time()
    return end_time - start_time

@profile
def test_pillow_open(AB_path):
    start_time = time.time()
    AB = Image.open(AB_path).convert("RGB")
    w, h = AB.size
    if w > h:
        w2 = int(w / 3)
        A = AB.crop((0, 0, w2, h))
        B1 = AB.crop((w2, 0, w2 * 2, h))
        B2 = AB.crop((w2 * 2, 0, w, h))
    else:
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B1 = AB.crop((w2, 0, w, h))
        B2 = B1
    end_time = time.time()
    return end_time - start_time

@profile
def test_pillow_open(AB_path):
    start_time = time.time()
    AB = Image.open(AB_path).convert("RGB")
    w, h = AB.size
    if w > h:
        w2 = int(w / 3)
        A = AB.crop((0, 0, w2, h))
        B1 = AB.crop((w2, 0, w2 * 2, h))
        B2 = AB.crop((w2 * 2, 0, w, h))
    else:
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B1 = AB.crop((w2, 0, w, h))
        B2 = B1
    end_time = time.time()
    return end_time - start_time


@profile
def test_cv2_open(AB_path):
    start_time = time.time()
    
    # Read the image using cv2
    AB = cv2.imread(AB_path)
    
    # Convert the image from BGR to RGB
    AB = cv2.cvtColor(AB, cv2.COLOR_BGR2RGB)
    
    # Get the dimensions of the image
    h, w, _ = AB.shape
    
    # Crop the image into A, B1, and B2
    if w > h:
        w2 = int(w / 3)
        A = AB[:, :w2, :]
        B1 = AB[:, w2:w2 * 2, :]
        B2 = AB[:, w2 * 2:, :]
    else:
        w2 = int(w / 2)
        A = AB[:, :w2, :]
        B1 = AB[:, w2:, :]
        B2 = B1  # In this case, B2 is the same as B1
    
    end_time = time.time()
    return end_time - start_time


@profile
def test_pyspng_open(AB_path):
    start_time = time.time()
    
    # Open and read the image using pyspng
    with open(AB_path, 'rb') as f:
        image_data = f.read()
        AB = pyspng.load(image_data)
    
    # Convert the image to RGB if necessary (pyspng loads as RGB by default)
    h, w, _ = AB.shape
    
    # Split the image into A and B
    if w > h:
        w2 = int(w / 3)
        A = AB[:, :w2, :]
        B1 = AB[:, w2:w2 * 2, :]
        B2 = AB[:, w2 * 2:, :]
    else:
        w2 = int(w / 2)
        A = AB[:, :w2, :]
        B1 = AB[:, w2:, :]
        B2 = B1  # In this case, B2 is the same as B1
    
    end_time = time.time()
    return end_time - start_time, A, B1, B2

def list_image_files(base_dir, pattern="*.png"):
    return glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)

def load_with_pyspng(file_path):
    with open(file_path, "rb") as f:
        image = pyspng.load(f.read())
    return image

def load_with_pillow_simd(file_path):
    image = Image.open(file_path).convert("RGB")
    return image

def load_with_torchvision(file_path):
    image = io.read_image(file_path, mode=io.ImageReadMode.RGB)
    return image

def load_with_cv2(file_path):
    image = cv2.imread(file_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_image(image):
    # Example preprocessing logic
    # Adjust this based on your specific needs
    w, h = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
    if w > h:
        w2 = int(w / 3)
        A = image.crop((0, 0, w2, h)) if isinstance(image, Image.Image) else image[:, :w2]
        B1 = image.crop((w2, 0, w2 * 2, h)) if isinstance(image, Image.Image) else image[:, w2:w2*2]
        B2 = image.crop((w2 * 2, 0, w, h)) if isinstance(image, Image.Image) else image[:, w2*2:]
    else:
        w2 = int(w / 2)
        A = image.crop((0, 0, w2, h)) if isinstance(image, Image.Image) else image[:, :w2]
        B1 = image.crop((w2, 0, w, h)) if isinstance(image, Image.Image) else image[:, w2:]
        B2 = B1
    return A, B1, B2

def benchmark_function(load_func, file_list):
    total_time = 0
    for file_path in file_list:
        start_time = time.time()
        image = load_func(file_path)
        preprocess_image(image)
        end_time = time.time()
        total_time += (end_time - start_time)
    return total_time / len(file_list)

base_directory = "/Users/linh/Downloads/segmentation/Su_HE/Panda"
image_files = list_image_files(base_directory)

torchvision_time = benchmark_function(load_with_torchvision, image_files)
print(f"Average time with pyspng: {torchvision_time:.6f} seconds")

# cv2_time = benchmark_function(load_with_cv2, image_files)
# print(f"Average time with pillow-simd: {cv2_time:.6f} seconds")

# AB_path = '/Users/linh/Downloads/13.jpg'
# torchvision_time = test_torchvision_read_image(AB_path)
# pillow_time = test_pillow_open(AB_path)
# cv2_time = test_cv2_open(AB_path)
# pyspng_time = test_pyspng_open(AB_path)

# print(f"Torchvision execution time: {torchvision_time:.6f} seconds")
# print(f"Pillow execution time: {pillow_time:.6f} seconds")
# print(f"CV2 execution time: {pyspng_time:.6f} seconds")


# run with this command line
# python -m memory_profiler test_perf_torchvision_PIL.py
# or use hyperfine
# hyperfine --runs 10 --warmup 3 --export-json results_PIL.json 'python test_perf_torchvision_PIL.py' --show-output