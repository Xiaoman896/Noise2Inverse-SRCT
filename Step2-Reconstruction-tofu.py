import numpy as np
import os, argparse
from scipy.ndimage import shift
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import subprocess
import time
from datetime import datetime
from PIL import Image
run = 1
starttime = datetime.now()

# Path to directory containing TIFF images
parser = argparse.ArgumentParser(description='Reconstruction with tofu ez')
parser.add_argument('-raw_dir', type=str, default='/root', help='Path to split datasets')
parser.add_argument('-y_start', type=int, default=0, help='position along y-axis of the projection to start CT reconstruction')
parser.add_argument('-y_thick', type=int, default=0, help='how many lines along y-axis of projection to reconstruct')
parser.add_argument('-y_step', type=int, default=1, help='how intermediate lines along y-axis of projection to reconstruct')
parser.add_argument('-flag_PhR', type=bool, default=False, help='Phase retrieval used, True or False')
parser.add_argument('-energy', type=float, default=30, help='beam energy in keV')
parser.add_argument('-distance', type=float, default=1.5, help='sample to detector distance in meters')
parser.add_argument('-pixelsize', type=float, default=13e-6, help='pixelsize in meters')
parser.add_argument('-deltabeta', type=float, default=200, help='delta over beta ratio')
parser.add_argument('-Ring_removal', type=bool, default=False, help='Ring artifacts removal, True: enable; False:disable')
parser.add_argument('-h_sigma', type=int, default=3, help='Ring removal attribute horizontal sigma')
parser.add_argument('-v_sigma', type=int, default=1, help='Ring removal attribute vertical sigma')
parser.add_argument('-CoR_auto', type=bool, default=True, help='Center of rotation - automatic calculation, True: enable; False:disable')
parser.add_argument('-CoR_manu', type=float, default=1028, help='Center of rotation - manual defination, CoR_manu only works when CoR_auto is False')
parser.add_argument('-Delete_temp', type=bool, default=True, help='Delete temporary folder after')
args, unparsed = parser.parse_known_args()

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

splits = sorted([f for f in os.listdir(os.path.join(args.raw_dir)) if os.path.isdir(os.path.join(args.raw_dir,f))])
for i in range(len(splits)):

	# relavant paths
	PATH = os.path.join(args.raw_dir,splits[i])
	SAVE = PATH + "/sli"
	TEMP = os.path.join(args.raw_dir,"_temp")

	# projection values
	files = sorted([f for f in os.listdir(os.path.join(PATH,"tomo")) if os.path.isfile(os.path.join(PATH,"tomo",f))])
	number = len(files)
	im_start = Image.open(os.path.join(PATH,"tomo",files[0]))
	width, height = im_start.size

	# find the center of rotation automatically
	if args.CoR_auto:
		im_end = Image.open(os.path.join(PATH,"tomo/",files[-1]))
		CoR = find_rotation_axis(im_start, im_end)
		print(f"Automatically calculated CoR is: "+str(CoR))
	else:
		CoR = args.CoR_manu
		print(f"CoR is manually defined: "+str(CoR))

	# ring removal
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
	h_sigma = args.h_sigma
	v_sigma = args.v_sigma

	# phase retreival
	energy = args.energy
	distance = args.distance
	pixelsize = args.pixelsize
	deltabeta = args.deltabeta
	regrate = 0.4339*np.log(args.deltabeta)+0.0034

	# reconstruction y-position
	y_start = args.y_start
	if args.y_thick == 0:
		y_thick = height
	else:
		y_thick = args.y_thick
	y_step = args.y_step
	y_all = np.arange(np.ceil(-height/2),np.ceil(height/2+1),1)
	regionstart = y_all[y_start]
	regionend = y_all[y_start+y_thick]
	if args.Ring_removal == True:
		newregion = len(np.arange(regionstart,regionend,y_step))
		y_all = np.arange(np.ceil(-newregion/2),np.ceil(newregion/2+1),1)
		regionstart = y_all[0]
		regionend = y_all[0+newregion]
			
	if args.Ring_removal == False:
		# Reconstruct without Ring Removal or Phase Retrieval
		if args.flag_PhR == False:
			os.system("tofu reco --overall-angle 180 --flats "+os.path.join(PATH,"flats")+" --darks "+os.path.join(PATH,"darks")+" --projections "+os.path.join(PATH,"tomo")+" --output "+os.path.join(SAVE,"sli")+" --absorptivity --fix-nan-and-inf --center-position-x "+str(CoR)+" --number "+str(number)+" --volume-angle-z 0.00000 --region="+str(regionstart)+","+str(regionend)+","+str(y_step)+" --output-bytes-per-file 0 --slice-memory-coeff=0.7")

		# Reconstruct wihtout Ring Removal but with Phase Retreival
		else:
			os.system("tofu reco --overall-angle 180 --flats "+os.path.join(PATH,"flats")+" --darks "+os.path.join(PATH,"darks")+" --projections "+os.path.join(PATH,"tomo")+" --output "+os.path.join(SAVE,"sli")+" --fix-nan-and-inf --disable-projection-crop --delta 1e-6 --energy "+str(energy)+" --propagation-distance "+str(distance)+" --pixel-size "+str(pixelsize)+" --regularization-rate "+str(regrate)+" --center-position-x "+str(CoR)+" --number "+str(number)+" --volume-angle-z 0.00000 --region="+str(regionstart)+","+str(regionend)+","+str(y_step)+" --output-bytes-per-file 0 --slice-memory-coeff=0.7")

	else:
		# Reconstruct with Ring Removal but without Phase Retrieval
		if args.flag_PhR == False:
			os.system("tofu sinos --flats "+os.path.join(PATH,"flats")+" --darks "+os.path.join(PATH,"darks")+" --projections "+os.path.join(PATH,"tomo")+" --output "+os.path.join(TEMP,"sinos/sino-%04i.tif")+" --absorptivity --fix-nan-and-inf --number "+str(number)+" --y "+str(y_start)+" --height "+str(y_thick)+" --y-step "+str(y_step)+" --output-bytes-per-file 0")

		# Reconstruct with Ring Removal and Phase Retrieval
		else:
			os.system("tofu preprocess --fix-nan-and-inf --projection-filter none --delta 1e-6 --flats "+os.path.join(PATH,"flats")+" --darks "+os.path.join(PATH,"darks")+" --projections "+os.path.join(PATH,"tomo")+" --output "+os.path.join(TEMP,"proj-step1/proj-%04i.tif")+" --energy "+str(energy)+" --propagation-distance "+str(distance)+" --pixel-size "+str(pixelsize)+" --regularization-rate "+str(regrate)+" --output-bytes-per-file 0")
			os.system("tofu sinos --projections "+os.path.join(TEMP,"proj-step1")+" --output "+os.path.join(TEMP,"sinos/sino-%04i.tif")+" --number "+str(number)+" --y "+str(y_start)+" --height "+str(y_thick)+" --y-step "+str(y_step)+" --output-bytes-per-file 0")

		os.system('ufo-launch read path='+os.path.join(TEMP,'sinos')+' ! pad x='+str(padx)+' width='+str(padwidth)+' y='+str(pady)+' height='+str(padheight)+' addressing-mode=mirrored_repeat ! fft dimensions=2 ! filter-stripes horizontal-sigma='+str(h_sigma)+' vertical-sigma='+str(v_sigma)+' ! ifft dimensions=2 crop-width='+str(padwidth)+' crop-height='+str(padheight)+' ! crop x='+str(padx)+' width='+str(width)+' y='+str(pady)+' height='+str(number)+' ! write filename="'+os.path.join(TEMP,'sinos-filt/sin-%04i.tif')+'" bytes-per-file=0 tiff-bigtiff=False')
		os.system("tofu sinos --projections "+os.path.join(TEMP,"sinos-filt")+" --output "+os.path.join(TEMP,"proj-step2/proj-%04i.tif")+" --number "+str(newregion)+" --output-bytes-per-file 0")
		os.system("tofu reco --overall-angle 180 --projections "+os.path.join(TEMP,"proj-step2")+" --output "+os.path.join(SAVE,"sli")+" --center-position-x "+str(CoR)+" --number "+str(number)+" --volume-angle-z 0.00000 --region="+str(regionstart)+","+str(regionend)+",1 --output-bytes-per-file 0 --slice-memory-coeff=0.7")

	# Remove temporary folder
	if args.Delete_temp == True:
		if os.path.isdir(TEMP):
			os.system("rm -r "+TEMP)

elapsedtime = str(datetime.now() - starttime)
print("Finisehd in " + elapsedtime)
starttime = datetime.now()



