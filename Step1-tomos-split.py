import os
import subprocess
import numpy as np
import shutil
import sys
from datetime import datetime
run = 1
starttime = datetime.now()
totaltime = datetime.now()
present_dir = os.getcwd()

def root_path():
	return os.path.abspath(os.sep)

def progressbar(it, prefix="", size=50, out=sys.stdout):
	count = len(it)
	def show(j):
		x = int(size*j/count)
		out.write("%s[%s%s] %i/%i\r" % (prefix, u"#"*x, "."*(size-x), j, count))
		out.flush()        
	show(0)
	for i, item in enumerate(it):
		yield item
		show(i+1)
	out.write("\n")
	out.flush()

# Path to directory containing TIFF images
parser = argparse.ArgumentParser(description='Split the tomos for sub-reconstruction')
parser.add_argument('-image_raw_path', type=str, required = True, help='Path including flats, darks, and tomo folders')
parser.add_argument('-image_save_path', type=str, required = True, help='Path for saving the splitted tomo files')
parser.add_argument('-m', type=int, default=2, help='Number of splits')
args, unparsed = parser.parse_known_args()

PATH = args.image_raw_path
SAVE = args.image_save_path
splits = args.m

# run only if tomo darks and flats folders are present
if os.path.isdir(PATH+"/tomo")+os.path.isdir(PATH+"/flats")+os.path.isdir(PATH+"/darks") == 3:
	if not os.path.isdir(SAVE):
		os.mkdir(SAVE)

	# get number of files inside the desired directory
	files = sorted([f for f in os.listdir(os.path.join(PATH,"tomo")) if os.path.isfile(os.path.join(PATH,"tomo",f))])
	filesnum = np.size(files)

	for i in range(splits):
		# make output folder
		if not os.path.isdir(os.path.join(SAVE,str(i).zfill(4))):
			os.mkdir(os.path.join(SAVE,str(i).zfill(4)))
			os.mkdir(os.path.join(SAVE,str(i).zfill(4),"tomo"))
		# copy every i-th file
		# for j in np.arange(i,filesnum,splits):
		for j in progressbar(np.arange(i,filesnum,splits), "Making split #"+str(i)+": "):
			# print(files[j])
			shutil.copy(os.path.join(PATH,"tomo",files[j]), os.path.join(SAVE,str(i).zfill(4),"tomo",files[j]))
		os.system("cp -r "+PATH+"/flats "+os.path.join(SAVE,str(i).zfill(4),"flats"))
		os.system("cp -r "+PATH+"/darks "+os.path.join(SAVE,str(i).zfill(4),"darks"))

# otherwise notify that raw data is incomplete
else:
	print("Incomplete raw data folder")

elapsedtime = str(datetime.now() - starttime)
print("Finisehd in " + elapsedtime)
starttime = datetime.now()
