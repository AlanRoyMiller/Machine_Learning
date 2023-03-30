import PIL, PIL.Image
import os
import glob
import shutil
import numpy as np
import hashlib
import logging


def validate_images(input_dir: str, output_dir: str, log_file="log_file.log", formatter: str = "07d"):
    if not os.path.exists(os.path.abspath(input_dir)):
        raise ValueError

    directory_list = sorted([file for file in glob.glob(f"{input_dir}/**", recursive=True) if os.path.isfile(file)]) # return sorted list of all files in input_dir

    os.makedirs(output_dir, exist_ok=True)

    # Variables
    valid_filetypes = (".jpg", ".JPG", ".jpeg", ".JPEG")
    output_filetype = ".jpg"
    max_filesize = 250000
    min_width, min_height = 100, 100
    image_modes = ("RGB", "L")
    variance_threshold = 0
    hash_values = []
    copied_files = 0

    # Create log file
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG, format='%(message)s')

    for file in directory_list:

        filename_new = file.removeprefix(input_dir)

        if not file.endswith(valid_filetypes):  # Correct File ending
            logging.debug(f"{filename_new}, 1\n")
            continue

        elif os.path.getsize(file) > max_filesize:  # The file size does not exceed max_filesize
            logging.debug(f"{filename_new}, 2\n")
            continue

        try:
            img = PIL.Image.open(file)
            if img.mode not in image_modes:  # Mode is in image_modes
                logging.debug(f"{filename_new}, 4\n")
                continue

            elif (img.width < min_width) and (
                    img.height < min_height):  # Check if height and width are bigger than the maximum allowed
                logging.debug(f"{filename_new}, 4\n")
                continue

            image_array = np.array(img)
            image_variance = np.var(image_array)
            if image_variance < variance_threshold:  # Check if image data has a variance larger than the threshold
                logging.debug(f"{filename_new}, 5\n")
                continue

            pic_hash = hashlib.sha256(img.tobytes()).hexdigest()
            if pic_hash in hash_values:  # Check if same image has not been copied already.
                logging.debug(f"{filename_new}, 6\n")
                continue
            hash_values.append(pic_hash)

        except PIL.UnidentifiedImageError:  # Check if file can be read as image
            logging.debug(f"{filename_new}, 3\n")

        else:
            new_filename = output_dir + f"/{copied_files:{formatter}}{output_filetype}"

            shutil.copy(file, new_filename)
            copied_files += 1

    return copied_files
