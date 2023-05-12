import os, argparse, shutil
import numpy as np
from PIL import Image

# Path to directory containing TIFF images
parser = argparse.ArgumentParser(description='Split the tomos for sub-reconstruction')
parser.add_argument('-image_raw_path', type=str, required = True,
                    help='Path including tomo/ flats/ darks/ folder')
parser.add_argument('-image_save_path', type=str, required = True,
                    help='Path for saving the splitted tomo files')
parser.add_argument('-m', type=int, default=2, help='Number of sections')

args, unparsed = parser.parse_known_args()
# Number of sections to split the images into
# create the output directories
for i in range(1, args.m+1):
    os.makedirs(os.path.join(args.image_save_path, f"section_{i}/tomo/"), exist_ok=False)
    os.makedirs(os.path.join(args.image_save_path, f"section_{i}/flats/"), exist_ok=False)
    os.makedirs(os.path.join(args.image_save_path, f"section_{i}/darks/"), exist_ok=False)

# get the tomo list of tiff files in the input directory
tomo_files = sorted([f for f in os.listdir(os.path.join(args.image_raw_path, f"tomo/")) if f.endswith(".tif")])
if len(tomo_files) == 0:
    print("No tif files in the tomo folder, please check your path")

# split the images into sections and save them to the output directories
for i, file_name in enumerate(tomo_files):
    print(f'Now processing tomo index: {i}/{len(tomo_files)-1}')
    section_num = (i % args.m) + 1
    output_path = os.path.join(args.image_save_path, f"section_{section_num}/tomo/", file_name)
    img = Image.open(os.path.join(args.image_raw_path, f"tomo/", file_name))
    img.save(output_path)

# get the flats list of tiff files in the input directory
flats_files = sorted([f for f in os.listdir(os.path.join(args.image_raw_path, f"flats/")) if f.endswith(".tif")])
if len(flats_files) == 0:
    print("No tif files in the flat folder, please check your path")

for i in range(1, args.m+1):
    for f in flats_files:
        source_file = os.path.join(args.image_raw_path, f"flats/",f)
        dest_file = os.path.join(args.image_save_path, f"section_{i}/flats/",f)
        shutil.copyfile(source_file, dest_file)

# get the darks list of tiff files in the input directory
darks_files = sorted([f for f in os.listdir(os.path.join(args.image_raw_path, f"darks/")) if f.endswith(".tif")])
if len(darks_files) == 0:
    print("No tif files in the flat folder, please check your path")

for i in range(1, args.m+1):
    for f in darks_files:
        source_file = os.path.join(args.image_raw_path, f"darks/",f)
        dest_file = os.path.join(args.image_save_path, f"section_{i}/darks/",f)
        shutil.copyfile(source_file, dest_file)
