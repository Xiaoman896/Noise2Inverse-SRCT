import numpy as np
import os
import sys
import numpy as np
import PIL
import re

def root_path():
	return os.path.abspath(os.sep)

def progressbar2(it, prefix="", size=50, out=sys.stdout):
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

def read_images(image_directory):
	# This function takes the directory containing darks, flats, and tomo folder 
	# then returns 3D arrays for darks, flats, and tomo

	# get names of all files inside directory
	directory_file_names = sorted_nicely([f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))])

	# get the number of images in folder
	directory_size = np.size(directory_file_names)
		
	# open the first image
	image_file = PIL.Image.open(os.path.join(image_directory, directory_file_names[0]))
	
	# change from image to array
	image_as_array = np.array(image_file)
	
	# get the shape of the array
	image_as_array_shape = np.shape(image_as_array)
	
	# preallocate space
	preallocate_matrix = np.zeros((directory_size, image_as_array_shape[0], image_as_array_shape[1]))

	# open all images and save as 3d array
	for i in range(directory_size):
		image_file = PIL.Image.open(os.path.join(image_directory, directory_file_names[i]))
		preallocate_matrix[i] = np.array(image_file)
	image_stack_array = preallocate_matrix

	return image_stack_array


def delete_position(read_stack, slice_thickness, slice_position):
	# This function reduces the 3d array stack of images, to just the specified slices for 
	# which reconstruction will be performed

	# Get dimensions of projections
	num_projections, num_height, num_width = np.shape(read_stack)

	vector = np.arange(slice_thickness)+slice_position

	delete_vector = np.arange(num_height)
	delete_vector = np.delete(delete_vector, vector)

	# Preallocate
	preallocate_matrix = np.zeros((num_projections, slice_thickness, num_width))
	
	for i in range(num_projections):
		stack = read_stack[i,:,:]
		stack = np.delete(stack, delete_vector, 0)
		preallocate_matrix[i,:,:] = stack

	return preallocate_matrix


def sorted_nicely( l ):
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)
