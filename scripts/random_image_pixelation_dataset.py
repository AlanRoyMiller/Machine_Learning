import glob
from PIL import Image
import numpy as np
from Normalized_grayscale import to_grayscale
from Pixelate_area import prepare_image
from torch.utils.data import Dataset



class RandomImagePixelationDataset(Dataset):

    def __init__(self, image_dir, width_range, height_range, size_range, dtype=None):

        if any((tup[0] < 2) for tup in (width_range, height_range, size_range)) or \
                any(tup[0] > tup[1] for tup in (width_range, height_range, size_range)):

            raise ValueError

        self.image_dir = sorted(glob.glob(f"{image_dir}/**/*.jpg", recursive=True))
        self.dtype = dtype
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range


    def __getitem__(self, index):
        with Image.open(self.image_dir[index]) as image_file:
            image_file = np.array(image_file, dtype=self.dtype)
        image_file = to_grayscale(image_file)
        rng = np.random.default_rng(seed=index)

        width = rng.integers(low=self.width_range[0], high=self.width_range[1])
        height = rng.integers(low=self.height_range[0], high=self.height_range[1])
        size = rng.integers(low=self.size_range[0], high=self.size_range[1])
        x = rng.integers(low=0, high=image_file.shape[2] - width)
        y = rng.integers(low=0, high=image_file.shape[1] - height)

        pixelated_image, known_array, target_array,  = prepare_image(image_file, x, y, width, height, size)


        return pixelated_image, known_array, target_array, self.image_dir[index]

    def __len__(self):
        return len(self.image_dir)
