import imageio
import numpy as np
import os
import sys
from tensorflow.python.client import device_lib
from scipy.ndimage import shift

def save2img(d_img, fn):
	# _min, _max = d_img.min(), d_img.max()
	# if np.abs(_max - _min) < 1e-4:
	#     img = np.zeros(d_img.shape)
	# else:
	# img = (d_img - _min) * 255. / (_max - _min)
	img = d_img
	img = img.astype('uint8')
	imageio.imwrite(fn, img)

def scale2uint8(d_img):
	#     _min, _max = d_img.min(), d_img.max()
	np.nan_to_num(d_img, copy=False)
	_min, _max = np.percentile(d_img, 0.05), np.percentile(d_img, 99.95)
	s_img = d_img.clip(_min, _max)
	if _max == _min:
		s_img -= _max
	else:
		s_img = (s_img - _min) * 255. / (_max - _min)
	return s_img.astype('uint8')

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def find_rotation_axis(proj_start, proj_end):
	proj_start = np.array(proj_start)
	proj_end_mirr = np.flip(proj_end, axis=1)

	i = list(range(-50, 51, 1))
	shifted_img = np.array([shift(proj_end_mirr, (0, i), cval=np.mean(proj_end_mirr)) for i in i])
	I1 = shifted_img[:, :, proj_start.shape[1]//5:proj_start.shape[1]//5*4]
	I1 = np.float32(I1)
	I2 = proj_start[:, proj_start.shape[1]//5:proj_start.shape[1]//5*4]
	I2= np.float32(np.tile(I2[np.newaxis, ...], (shifted_img.shape[0], 1, 1)))
	L1 = np.sum(np.sum(abs(I1 - I2),axis=1),axis=1)
	mispixels = (np.argmin(L1) + min(i))/2

	rotation_axis = mispixels + proj_start.shape[-1]//2

	return rotation_axis
