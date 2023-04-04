import PIL.Image
import numpy as np
from PIL import Image



def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    new_image = pil_image / 255.0

    print(new_image.shape)

    c_linear = np.where((new_image <= 0.04045), (new_image / 12.92), (((new_image + 0.055) / 1.055) ** 2.4))

    y_linear = c_linear[:, :, 0] * 0.2126 + c_linear[:, :, 1] * 0.7152 + c_linear[:, :, 2] * 0.0722

    y_srgb = np.where((y_linear <= 0.0031308), (y_linear * 12.92), (((y_linear ** (1 / 2.4)) * 1.055) - 0.055))

    new_image = y_srgb * 255.0

    new_img = Image.fromarray(new_image.astype(np.uint8))

    new_img.save('new_image.jpg')


    #return ...

