import os
import argparse
import numpy as np
from datetime import datetime
import subprocess
run = 1
starttime = datetime.now()

def boolean_string(s):
	if s not in {'False', 'True'}:
		raise ValueError('Not a valid boolean string')
	return s == 'True'

parser = argparse.ArgumentParser(description='Inputs from raw data to N2I denoised reconstruction')
# Step 1 arguments
parser.add_argument('-PATH', type=str, required = True, help='Path including flats, darks, and tomo folders')
parser.add_argument('-SAVE', type=str, required = True, help='Path for saving the splitted tomo files')
parser.add_argument('-splits', type=int, default=2, help='Number of splits')
# Step 2 arguments
parser.add_argument('-y_start', type=int, default=0, help='position along y-axis of the projection to start CT reconstruction')
parser.add_argument('-y_thick', type=int, default=0, help='how many lines along y-axis of projection to reconstruct')
parser.add_argument('-y_step', type=int, default=1, help='how intermediate lines along y-axis of projection to reconstruct')
parser.add_argument('-PR', type=boolean_string, default=False, help='Phase retrieval used, True or False')
parser.add_argument('-PR_energy', type=float, default=30, help='beam energy in keV')
parser.add_argument('-PR_distance', type=float, default=1.5, help='sample to detector distance in meters')
parser.add_argument('-PR_pixelsize', type=float, default=13e-6, help='pixelsize in meters')
parser.add_argument('-PR_deltabeta', type=float, default=200, help='delta over beta ratio')
parser.add_argument('-RR', type=boolean_string, default=False, help='Ring artifacts removal, True: enable; False:disable')
parser.add_argument('-RR_h_sigma', type=int, default=3, help='Ring removal attribute horizontal sigma')
parser.add_argument('-RR_v_sigma', type=int, default=1, help='Ring removal attribute vertical sigma')
parser.add_argument('-CoR_auto', type=boolean_string, default=True, help='Center of rotation - automatic calculation, True: enable; False:disable')
parser.add_argument('-CoR_manu', type=float, default=1028, help='Center of rotation - manual defination, CoR_manu only works when CoR_auto is False')
parser.add_argument('-Delete_temp', type=boolean_string, default=True, help='Delete temporary folder after')
# Step 3 arguments
parser.add_argument('-exp_name', type=str, default='exp', help='Name of scaned sample')
parser.add_argument('-gv_255', type=boolean_string, default=True, help='Scale the grey value to 0-255')
parser.add_argument('-crop', type=boolean_string, default=False, help='Enable cropping by setting True')
parser.add_argument('-start_row', type=int, default=512, help='For cropping: started row index; It is only enable when crop = True')
parser.add_argument('-start_column', type=int, default=512, help='For cropping: started column index; It is only enable when crop = True')
parser.add_argument('-width', type=int, default=512, help='For cropping: Cropped region size (can be ); It is only enable when crop = True')
# Step 4 and 5 arguments
parser.add_argument('-gpus', type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName', type=str, default='exp', help='Experiment name')
parser.add_argument('-mse', type=boolean_string, default=False, help='True: use mse as loss function; False: use ssim as the loss function')
parser.add_argument('-lunet', type=int, default=4, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input depth (use for 3D CT image only)')
parser.add_argument('-psz', type=int, default=64, help='cropping patch size')
parser.add_argument('-mbsz', type=int, default=32, help='mini-batch size')
parser.add_argument('-epoch_save', type=int, default=100, help='Save the model and images for every epoch_save')
parser.add_argument('-maxiter', type=int, default=4000, help='maximum iterations')
parser.add_argument('-printout', type=boolean_string, default=True, help='1: print to terminal; 0: redirect to file')

args, unparsed = parser.parse_known_args()

envs = open("envs.txt", "r")
env1 = envs.readline().strip()
env2 = envs.readline().strip()

# Run Step 1
os.system("python Step1-tomos-split.py -image_raw_path "+args.PATH+" -image_save_path "+args.SAVE+" -m "+str(args.splits))

# Run Step 2
subprocess.call(". "+env1+" && python Step2-Reconstruction-tofu.py -raw_dir "+args.SAVE+" -y_start "+str(args.y_start)+" -y_thick "+str(args.y_thick)+" -y_step "+str(args.y_step)+" -flag_PhR "+str(args.PR)+" -energy "+str(args.PR_energy)+" -distance "+str(args.PR_distance)+" -pixelsize "+str(args.PR_pixelsize)+" -deltabeta "+str(args.PR_deltabeta)+" -Ring_removal "+str(args.RR)+" -h_sigma "+str(args.RR_h_sigma)+" -v_sigma "+str(args.RR_v_sigma)+" -CoR_auto "+str(args.CoR_auto)+" -CoR_manu "+str(args.CoR_manu)+" -Delete_temp "+str(args.Delete_temp), shell=True)

# Run Step 3
subprocess.call(". "+env2+" && python Step3-Dataset_preparation.py -exp_name "+args.exp_name+" -sli_dir "+args.SAVE+" -dataset_dir "+args.SAVE+" -gv_255 "+str(args.gv_255)+" -crop "+str(args.crop)+" -start_row "+str(args.start_row)+" -start_column "+str(args.start_column)+" -width "+str(args.width)+" -m "+str(args.splits), shell=True)

# Run Step 4
for f in os.listdir(args.SAVE):
	if f.endswith(".h5"):
		h5fn = f
subprocess.call(". "+env2+" && python Step4-N2I-main.py -gpus "+args.gpus+" -expName "+args.expName+" -mse "+str(args.mse)+" -lunet "+str(args.lunet)+" -depth "+str(args.depth)+" -psz "+str(args.psz)+" -mbsz "+str(args.mbsz)+" -epoch_save "+str(args.epoch_save)+" -maxiter "+str(args.maxiter)+" -dsfn "+args.SAVE+" -h5fn "+h5fn+" -print "+str(args.printout), shell=True)

# Run Step 5
mdfn = os.path.splitext(h5fn)[0]
if args.mse == True:
	mdfn = mdfn+"_mse_output"
	allfiles = sorted([f for f in os.listdir(os.path.join("Output",mdfn)) if f.endswith(".h5")])
else:
	mdfn = mdfn+"_ssim_output"
	allfiles = sorted([f for f in os.listdir(os.path.join("Output",mdfn)) if f.endswith(".h5")])
mdl = allfiles[len(allfiles)-1]
subprocess.call(". "+env2+" && python Step5-N2I-infer.py -gpus "+args.gpus+" -mdfn "+mdfn+" -mdl "+mdl+" -dsfn "+args.SAVE+" -depth "+str(args.depth)+" -h5fn "+h5fn, shell=True)


