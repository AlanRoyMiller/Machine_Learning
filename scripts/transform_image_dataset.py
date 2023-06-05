from random_augmentation import random_augmented_image
from torchvision import transforms
from PIL import Image
import torch
import glob
import os


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir):
        self.jpg_files = sorted(glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True))

    def __getitem__(self, index: int):
        img_path = self.jpg_files[index]
        img = Image.open(img_path)
        return img, index

    def __len__(self):
        return len(self.jpg_files)



class TransformedImageDataset(torch.utils.data.Dataset):
    def  __init__(self, dataset: ImageDataset, image_size):
        self.dataset = dataset
        self.image_size = image_size

    def __getitem__(self, index: int):
        img = self.dataset.__getitem__(index)[0]
        transformed_img = random_augmented_image(img, self.image_size, index)
        return transformed_img, index

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    images_directory = r"path to images"
    imgs = ImageDataset(image_dir=images_directory)
    transformed_imgs = TransformedImageDataset(imgs, image_size=300)

    for (original_img, index), (transformed_img, _) in zip(imgs, transformed_imgs):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original_img)
        axes[0].set_title("Original image")
        axes[1].imshow(transforms.functional.to_pil_image(transformed_img))
        axes[1].set_title("Transformed image")
        fig.suptitle(f"Image {index}")
        fig.tight_layout()
        plt.show()
