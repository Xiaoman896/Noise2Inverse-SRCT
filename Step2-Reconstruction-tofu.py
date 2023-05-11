import numpy as np
import os, argparse
from scipy.ndimage import shift
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import subprocess
import time
from datetime import datetime
from PIL import Image

# Path to directory containing TIFF images
parser = argparse.ArgumentParser(description='Reconstruction with tofu ez')
parser.add_argument('-raw_dir', type=str, default='/staff/duanx/Desktop/Network_Data_projects/prj35G12338/BMIT/rec/2022-5-27-ID/Xiaoman/Different_CL/4Al/15m/split/section_2/',
                    help='Path including tomo/ flats/ darks/ folder')
parser.add_argument('-flag_PhR', type=bool, default=False, help='Phase retrieval used, True or False')
parser.add_argument('-CoR_auto', type=bool, default=True, help='Center of rotation - automatic calculation, True: enable; False:disable')
parser.add_argument('-CoR_manu', type=float, default=1028, help='Center of rotation - manual defination, CoR_manu only works when CoR_auto is False')
parser.add_argument('-Ring_removal', type=bool, default=True, help='Ring artifacts removal, True: enable; False:disable')

args, unparsed = parser.parse_known_args()

starttime = datetime.now()
present_dir = args.raw_dir

def find_rotation_axis(proj_start, proj_end):
    proj_start = np.array(proj_start)
    proj_end_mirr = np.flip(proj_end, axis=1)

    i = list(range(-50, 51, 1))
    shifted_img = np.array([shift(proj_end_mirr, (0, i), cval=np.mean(proj_end_mirr)) for i in i])
    I1 = shifted_img[:, :, proj_start.shape[1]//5:proj_start.shape[1]//5*4] ## Get rid of the influence of the edges
    I1 = np.float32(I1)
    I2 = proj_start[:, proj_start.shape[1]//5:proj_start.shape[1]//5*4] ## Get rid of the influence of the edges
    I2= np.float32(np.tile(I2[np.newaxis, ...], (shifted_img.shape[0], 1, 1)))
    L1 = np.sum(np.sum(abs(I1 - I2),axis=1),axis=1)
    mispixels = (np.argmin(L1) + min(i))/2

    rotation_axis = mispixels + proj_start.shape[-1]//2

    return rotation_axis

################## INPUTS ##################

# relavant paths
PATH = present_dir
SAVE = present_dir + "sli"
TEMP = present_dir + "_temp"

# projection values
files = sorted(
    [f for f in os.listdir(os.path.join(present_dir,f"tomo/")) if os.path.isfile(os.path.join(present_dir,f"tomo/", f))])
number = len(files)
im_start = Image.open(os.path.join(present_dir,f"tomo/", files[0]))
width, height = im_start.size

# find the center of rotation automatically
if args.CoR_auto:
    im_end = Image.open(os.path.join(present_dir,f"tomo/", files[-1]))
    CoR = find_rotation_axis(im_start, im_end)
    print(f"Automatically calculated CoR is: {CoR}")
else:
    CoR = args.CoR_manu
    print(f"CoR is manually defined: {CoR}")

# projection padding
padwidth = 2
padheight = 2
count = 1
while padwidth <= width:
    padwidth = np.power(2,count)
    count = count+1
count = 1
while padheight <= number:
    padheight = np.power(2,count)
    count = count+1
padx = (padwidth-width)/2
pady = (padheight-number)/2


# phase retreival
energy = 30 # in keV
distance = 1.5 # in meters
pixelsize = 13e-6 # in microns
deltabeta = 2000
regrate = 0.4339*np.log(deltabeta)+0.0034

# reconstruction y-position
y_start = 0
y_thick = height
y_step = 1
y_all = np.arange(np.ceil(-height/2),np.ceil(height/2+1),y_step)
y_some = np.arange(np.ceil(-y_thick/2),np.ceil(y_thick/2+1),y_step)
regionstart = y_all[y_start]
regionend = y_all[y_start+y_thick]
################## COMMANDS ##################
#
if args.Ring_removal == False:

    if args.flag_PhR == False: ##only flat-dark correction
        os.system("tofu flatcorrect --flats " + PATH + "flats/ --darks " + PATH+ "darks/ --projections " + PATH + "tomo/ --output " +
            PATH + "tomo-FD/proj_%04i.tif --fix-nan-and-inf --output-bytes-per-file 1 --absorptivity")

        os.system("tofu tomo --projections " + PATH + "tomo-FD/ --output " + PATH +
                  "sli/sli_%05i.tif --fix-nan-and-inf --output-bytes-per-file 1 --axis " + str(CoR))
    else:
        os.system("tofu preprocess --fix-nan-and-inf --projection-filter none --delta 1e-6 --darks " + PATH + "darks --flats " + PATH +
            "flats --projections " + PATH + "tomo --output " + PATH + "tomo-FD-PhR/proj-%04i.tif --energy " + str(energy) +
                  "--retrieval-padded-height " + str(padheight) + " --retrieval-padded-width " + str(padwidth) +
            " --propagation-distance " + str(distance) + " --pixel-size " + str(pixelsize) + " --regularization-rate " + str(regrate) + " --output-bytes-per-file 1")

        os.system("tofu tomo --projections " + PATH + "tomo-FD-PhR/ --output " + PATH +
                  "sli/sli_%05i.tif --fix-nan-and-inf --output-bytes-per-file 1 --axis " + str(CoR))

else:

    if args.flag_PhR == False:  ##only flat-dark correction
        os.system(
            "tofu flatcorrect --flats " + PATH + "flats/ --darks " + PATH + "darks/ --projections " + PATH + "tomo/ --output " +
            TEMP + "/proj-step1/proj-%04i.tif --fix-nan-and-inf --output-bytes-per-file 0 --absorptivity")
    else:
        os.system(
            "tofu preprocess --fix-nan-and-inf --projection-filter none --delta 1e-6 --darks " + PATH + "darks --flats " + PATH +
            "flats --projections " + PATH + "tomo --output " + TEMP + "/proj-step1/proj-%04i.tif --energy " + str(
                energy) + " --propagation-distance " + str(distance) + " --pixel-size " + str(
                pixelsize) + " --regularization-rate " + str(regrate) + " --output-bytes-per-file 0")
    os.system(
        "tofu sinos --projections " + TEMP + "/proj-step1 --output " + TEMP + "/sinos/sin-%04i.tif --number " + str(
            number) + " --height " + str(height) + " --output-bytes-per-file 0")

    os.system('ufo-launch read path=' + TEMP + '/sinos ! pad x=' + str(padx) + ' width=' + str(padwidth) + ' y=' + str(
        pady) + ' height=' + str(
        padheight) + ' addressing-mode=mirrored_repeat ! fft dimensions=2 ! filter-stripes horizontal-sigma=40 vertical-sigma=1 ! ifft dimensions=2 crop-width=' + str(
        padwidth) + ' crop-height=' + str(padheight) + ' ! crop x=' + str(padx) + ' width=' + str(
        padwidth) + ' y=' + str(pady) + ' height=' + str(
        number) + ' ! write filename="' + TEMP + '/sinos-filt/sin-%04i.tif" bytes-per-file=0 tiff-bigtiff=False')
    os.system(
        "tofu sinos --projections " + TEMP + "/sinos-filt --output " + TEMP + "/proj-step2/proj-%04i.tif --number " + str(
            height) + " --output-bytes-per-file 0")
    os.system(
        "tofu reco --overall-angle 180  --projections " + TEMP + "/proj-step2 --output " + PATH + "/sli_Ring_removal/sli --center-position-x " + str(
            CoR) + " --number " + str(number) + " --volume-angle-z 0.00000 --region=" + str(regionstart) + "," + str(
            regionend) + "," + str(y_step) + " --output-bytes-per-file 0 --slice-memory-coeff=0.7")

sli_files = sorted([f for f in os.listdir(os.path.join(PATH, f"sli_Ring_removal/")) if f.endswith(".tif")])
# crop the sli image
for i, file_name in enumerate(sli_files):
    print(f'Now processing sli index: {i}/{len(sli_files) - 1}')
    img = Image.open(os.path.join(PATH, "sli_Ring_removal/", file_name))
    img_crop = img.crop((img.size[0]//2-width//2, img.size[1]//2-width//2, img.size[0]//2+width//2, img.size[1]//2+width//2))
    img_crop.save(os.path.join(PATH, "sli_Ring_removal/", file_name))

elapsedtime = str(datetime.now() - starttime)
print("Finisehd in " + elapsedtime)
starttime = datetime.now()



