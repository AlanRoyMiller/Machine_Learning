import time
#rom Copy_Files import validate_images
# 
# 
input_path = r"C:\Users\armil\Documents\Uni\Programming in Python2\ex7\training"
output_path = r"C:\Users\armil\Desktop\Image Depixelation Project"
log_file = r"C:\Users\armil\Desktop\python\log_file.log"
# 
# 
# start_time = time.perf_counter()
# validate_images(input_path, output_path, log_file)
# end_time = time.perf_counter()
# 
# # Print the execution time
# print(f"Execution time for Validating and copying images: {end_time - start_time:.6f} seconds")




from torch.utils.data import DataLoader
from random_image_pixelation_dataset import RandomImagePixelationDataset
from stack_with_padding_collate import stack_with_padding

ds = RandomImagePixelationDataset(
    output_path,
    width_range=(4, 32),
    height_range=(4, 32),
    size_range=(4, 16)
)

start_time = time.perf_counter()
dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=stack_with_padding)
end_time = time.perf_counter()

# Print the execution time
print(f"Execution time for grayscale conversion and pixelating images: {end_time - start_time:.6f} seconds")

import pickle
import numpy as np

with open(r'scripts\pickle_file\test_set.pkl', 'rb') as f:
    test_data = pickle.load(f)

pixelated_images = test_data['pixelated_images']
known_arrays = test_data['known_arrays']

print(ds[0])
print(pixelated_images[0])




from training_loop_early_stopping import training_loop, plot_losses


from simple_network import SimpleNetwork
import torch

torch.random.manual_seed(0)
train_data, eval_data = dl, zip(pixelated_images, known_arrays)
network = SimpleNetwork(32, 128, 1)
train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100)
for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
    print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")
plot_losses(train_losses, eval_losses)