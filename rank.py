import os
from torchvision import datasets
from PIL import Image

# Create folder to save images
save_dir = "mnist_images"
os.makedirs(save_dir, exist_ok=True)

# Download MNIST dataset
mnist_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True
)

# Save first 50 images as JPG
for i in range(50):
    image, label = mnist_dataset[i]

    # Convert grayscale image to PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image.numpy(), mode='L')

    # Save image
    image_path = os.path.join(save_dir, f"mnist_{i}_label_{label}.jpg")
    image.save(image_path, "JPEG")

print(f"Saved 50 MNIST images to '{save_dir}' folder.")