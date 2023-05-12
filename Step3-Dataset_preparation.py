import h5py
import os, argparse
import numpy as np
from PIL import Image
import tifffile

# Set the path to the folder containing the TIFF files
parser = argparse.ArgumentParser(description='Dataset preparation for Noise2Inverse: h5 file generation')
parser.add_argument('-exp_name', type=str, default='scaffold', help='Name of scaned sample')
parser.add_argument('-sli_dir', type=str, default='/staff/duanx/Desktop/Network_Data_projects/prj35G12338/BMIT/rec/2022-5-27-ID/Xiaoman/Different_CL/4Al/15m/split/',
                    help='the path should be split folder containing section_1, ..., section_m subfolders')
parser.add_argument('-dataset_dir', type=str, default='Dataset',
                    help='the path for saving the h5 file')
parser.add_argument('-gv_255', type=bool, default=True, help='Scale the grey value to 0-255')
parser.add_argument('-crop', type=bool, default=False, help='Enable cropping by setting True')
parser.add_argument('-start_row', type=int, default=512, help='For cropping: started row index; It is only enable when crop = True')
parser.add_argument('-start_column', type=int, default=512, help='For cropping: started column index; It is only enable when crop = True')
parser.add_argument('-width', type=int, default=512, help='For cropping: Cropped region size (can be ); It is only enable when crop = True')
parser.add_argument('-m', type=int, default=2, help='Number of sections')
args, unparsed = parser.parse_known_args()
if os.path.isdir(args.dataset_dir):
    print(f"saving the h5 file in {args.dataset_dir}")
else:
    os.mkdir(args.dataset_dir) # to save temp output
    print(f"creating the folder {args.dataset_dir}!")

# Get the list of TIFF files in the folder section_1 and use the strategy of X:1 Noise2Inverse
sli_files_1 = sorted([f for f in os.listdir(os.path.join(args.sli_dir, f"section_1/sli_Ring_removal/")) if f.endswith(".tif")])
#
# Read TIFF files and split into sections : target, section_1
train_ns_target = []
test_ns_target = []
for i, file_name in enumerate(sli_files_1):
    print(f"Now processing image #{i}/{len(sli_files_1)} in section 1")
    with Image.open(os.path.join(args.sli_dir, f"section_1/sli_Ring_removal/", file_name)) as tiff:
        img = np.asarray(tiff)
        img = img.astype(float) # convert to float if required
        if args.crop:
            img = img[args.start_row:args.start_row+args.width, args.start_column:args.start_column+args.width]
    if i < len(sli_files_1)//2 - 2 or i >= len(sli_files_1)//2 + 3: ## the middle 5 slices as the test dataset to display
        train_ns_target.append(img)
    else:
        test_ns_target.append(img)



if args.gv_255:
    _min_train_ns_target = np.min(train_ns_target)
    _max_train_ns_target = np.max(train_ns_target)
    _min_test_ns_target = np.min(test_ns_target)
    _max_test_ns_target = np.max(test_ns_target)
    train_ns_target = [(img - _min_train_ns_target) * (255 / (_max_train_ns_target - _min_train_ns_target)) for img in train_ns_target]
    test_ns_target = [(img - _min_test_ns_target) * (255 / (_max_test_ns_target - _min_test_ns_target)) for img in test_ns_target]


# # Read TIFF files and split into sections : target, section_2, ..., section_m
train_ns_input = np.zeros(np.shape(train_ns_target))
test_ns_input = np.zeros(np.shape(test_ns_target))

for k in range(1, args.m):
    sli_files = sorted([f for f in os.listdir(os.path.join(args.sli_dir, f"section_{k+1}/sli_Ring_removal/")) if f.endswith(".tif")])

    # Read TIFF files and split into sections
    train_temp = []
    test_temp = []
    for i, file_name in enumerate(sli_files):
        print(f"Now processing image #{i}/{len(sli_files)} in section {k+1}")
        with Image.open(os.path.join(args.sli_dir, f"section_{k+1}/sli_Ring_removal/", file_name)) as tiff:
            img = np.asarray(tiff)
            img = img.astype(float) # convert to float if required
            if args.crop:
                img = img[args.start_row:args.start_row + args.width, args.start_column:args.start_column + args.width]
            if i < len(sli_files_1)//2 - 2 or i >= len(sli_files_1)//2 + 3: ## the middle 5 slices as the test dataset to display
                train_temp.append(img)
            else:
                test_temp.append(img)

    train_ns_input += train_temp
    test_ns_input += test_temp
## calculate the mean
train_ns_input /= (args.m-1)
test_ns_input /= (args.m-1)
args.width = np.shape(img)[0]
if args.gv_255:
    _min_train_ns_input = np.min(train_ns_input)
    _max_train_ns_input = np.max(train_ns_input)
    _min_test_ns_input = np.min(test_ns_input)
    _max_test_ns_input = np.max(test_ns_input)
    train_ns_input = [(img - _min_train_ns_input) * (255 / (_max_train_ns_input - _min_train_ns_input)) for img in train_ns_input]
    test_ns_input = [(img - _min_test_ns_input) * (255 / (_max_test_ns_input - _min_test_ns_input)) for img in test_ns_input]

    with h5py.File(f"{args.dataset_dir}/{len(sli_files_1)}train_N2I_{args.exp_name}_Ring_removal_size_{str(args.width)}.h5", "w") as fd:
        fd.create_dataset("train_ns", data=train_ns_input, dtype=np.uint8)
        fd.create_dataset("test_ns", data=test_ns_input, dtype=np.uint8)
        fd.create_dataset("train_gt", data=train_ns_target, dtype=np.uint8)
        fd.create_dataset("test_gt", data=test_ns_target, dtype=np.uint8)
else:
    with h5py.File(f"{args.dataset_dir}/{len(sli_files_1)}train_N2I_{args.exp_name}_Ring_removal_size_{str(args.width)}.h5", "w") as fd:
        fd.create_dataset("train_ns", data=train_ns_input, dtype=np.float)
        fd.create_dataset("test_ns", data=test_ns_input, dtype=np.float)
        fd.create_dataset("train_gt", data=train_ns_target, dtype=np.float)
        fd.create_dataset("test_gt", data=test_ns_target, dtype=np.float)

print("HDF5 file created successfully!")

# make log folder and get log number
LOG = f"{args.dataset_dir}/{len(sli_files_1)}train_N2I_{args.exp_name}_Ring_removal_size_{str(args.width)}_log.txt"
# record log
loglines = ["exp_name = "+args.exp_name \
	   ,"sli_dir = "+args.sli_dir \
	   ,"dataset_dir = "+args.dataset_dir \
	   ,"gv_255 = "+str(args.gv_255) \
	   ,"crop = "+str(args.crop) \
	   ,"start_row = "+str(args.start_row) \
	   ,"start_column = "+str(args.start_column) \
	   ,"width = "+str(args.width) \
	   ,"m = "+str(args.m)]

with open(LOG, 'w') as f:
	f.write('\n'.join(loglines))
