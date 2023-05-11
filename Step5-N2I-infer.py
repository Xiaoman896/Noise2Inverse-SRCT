import tensorflow as tf
import numpy as np
import sys, os, time, argparse, shutil, scipy, h5py, glob
import tifffile

parser = argparse.ArgumentParser(description='predict the denoised images with trained model')
parser.add_argument('-gpus',  type=str, default="0,1", help='list of visiable GPUs')
parser.add_argument('-mdfn',  type=str, default='580train_N2I_scaffold_Ring_removal_size_512_ssim_output', help='folder (not the path) contains h5 dataset file')
parser.add_argument('-mdl', type=str, default='N2I-it03000.h5', help='model name')
parser.add_argument('-dsfn',  type=str, default='Dataset/', help='folder containing h5 test dataset file')
parser.add_argument('-depth', type=int, default=3, help='input depth (use for 3D CT image only)')
parser.add_argument('-h5fn',type=str, default='580train_N2I_scaffold_Ring_removal_size_512.h5', help='name of h5 test dataset file')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR

mdl = tf.keras.models.load_model("Output/" + args.mdfn + "/" + args.mdl)

args.dsfn = args.dsfn + args.h5fn
with h5py.File(args.dsfn, 'r') as h5fd:
    ns_img_test1 = h5fd['train_ns'][:]
    ns_img_test2 = h5fd['test_ns'][:]
    # gt_img_test = h5fd['test_gt'][:]
ns_img_test = np.concatenate((ns_img_test1, ns_img_test2))

# if len(ns_img_test.shape) == 3:
#     dn_img = mdl.predict(ns_img_test[:,:,:,np.newaxis]).squeeze()
# elif len(ns_img_test.shape) == 4:
#     dn_img = mdl.predict(ns_img_test).squeeze()
# else:
#     print("Model input must have N, H, W, C four dimension")
idx = [s_idx for s_idx in range(ns_img_test.shape[0] - args.depth)]
dn_img=[]
for s_idx in idx:
    X = np.array(np.transpose(ns_img_test[s_idx : (s_idx+args.depth)], (1, 2, 0)))
    X = X[np.newaxis, :, :, :]
    Y = mdl.predict(X[:1])
    dn = Y[0,:,:,0]
    dn_img.append(dn)
dn_img = np.array(dn_img)

itr_out_dir = os.path.join('Output/', os.path.dirname(args.h5fn), os.path.basename(args.h5fn).replace('.h5', '_predict_output'))
if os.path.isdir(itr_out_dir):
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save output

for i in range(dn_img.shape[0]):
    print(f'Now saving prediction slice index: {i}/{dn_img.shape[0]-1}')
    img = dn_img[i,:]
    tifffile.imwrite('%s/sli-%05d.tif' % (itr_out_dir, i), img)