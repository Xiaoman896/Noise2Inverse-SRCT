import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from util import save2img, str2bool
import sys, os, time, argparse, shutil, scipy.io, h5py, glob
from models import unet as make_generator_model           # import a generator model
from data import bkgdGen, gen_train_batch_bg, get1batch4test
import tifffile

parser = argparse.ArgumentParser(description='UNet, for noise/artifact removal')
parser.add_argument('-gpus',  type=str, default="0", help='list of visiable GPUs')
parser.add_argument('-expName', type=str, default='N2I', help='Experiment name')
parser.add_argument('-mse', type=bool, default=False, help='True: use mse as loss function; False: use ssim as the loss function')
parser.add_argument('-lunet', type=int, default=4, help='Unet layers')
parser.add_argument('-depth', type=int, default=3, help='input depth (use for 3D CT image only)')
parser.add_argument('-psz',   type=int, default=64, help='cropping patch size')
parser.add_argument('-mbsz',  type=int, default=32, help='mini-batch size')
parser.add_argument('-epoch_save',  type=int, default=100, help='Save the model and images for every epoch_save')
parser.add_argument('-maxiter', type=int, default=4000, help='maximum iterations')
parser.add_argument('-dsfn',  type=str, default='Dataset/', help='folder containing h5 dataset file')
parser.add_argument('-h5fn',  type=str, default='580train_N2I_scaffold_Ring_removal_size_512.h5', help='h5 dataset file')
parser.add_argument('-print', type=str2bool, default=True, help='1: print to terminal; 0: redirect to file')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)
os.mkdir('Output')
if len(args.gpus) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing INFO, WARNING, and ERROR
if args.mse:
    itr_out_dir = os.path.join('Output/', os.path.dirname(args.h5fn), os.path.basename(args.h5fn).replace('.h5', '_mse_output'))
else:
    itr_out_dir = os.path.join('Output/', os.path.dirname(args.h5fn), os.path.basename(args.h5fn).replace('.h5', '_ssim_output'))

if os.path.isdir(itr_out_dir):
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output

# redirect print to a file
if args.print == 0:
    sys.stdout = open('%s/%s' % (itr_out_dir, 'iter-prints.log'), 'w')

args.dsfn = args.dsfn + args.h5fn
# build minibatch data generator with prefetch
mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(
                                      dsfn=args.dsfn, mb_size=args.mbsz, \
                                      in_depth=args.depth, img_size=args.psz), \
                       max_prefetch=args.mbsz*4)

generator = make_generator_model(input_shape=(None, None, args.depth), nlayers=args.lunet)


gen_optimizer = tf.keras.optimizers.Adam(1*1e-4)

LossG=np.zeros((args.maxiter,1))
for epoch in range(args.maxiter):
    time_git_st = time.time()

    X_mb, y_mb = mb_data_iter.next() # with prefetch
    with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
        gen_tape.watch(generator.trainable_variables)

        gen_imgs = generator(X_mb, training=True)
        if args.mse:
            gen_loss = tf.losses.mean_squared_error(gen_imgs, y_mb)
        else:
        # loss_l1 = tf.losses.mean_absolute_error(gen_imgs, y_mb)
            gen_loss = 1-(tf.image.ssim(gen_imgs, tf.expand_dims(y_mb, axis=0), max_val=255))

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    LossG[epoch]=gen_loss.numpy().mean()
    scipy.io.savemat('%s/LossG.mat' % (itr_out_dir), {'LossG':np.array(LossG)})
    itr_prints_gen = '[Info] Epoch: %05d, gloss: %.4f, gen_elapse: %.4fs/itr' % ( epoch, gen_loss.numpy().mean(), (time.time() - time_git_st),)
    time_dit_st = time.time()

    print('%s' % itr_prints_gen)

    if epoch % (args.epoch_save) == 0:
        X222, y222 = get1batch4test(dsfn=args.dsfn, in_depth=args.depth)
        pred_img = generator.predict(X222[:1])

        tifffile.imwrite('%s/it%05d.tif' % (itr_out_dir, epoch), pred_img[0, :, :, 0])
        if epoch == 0:
            tifffile.imwrite('%s/gt.tif' % (itr_out_dir), y222[0, :, :, 0])
            tifffile.imwrite('%s/ns.tif' % (itr_out_dir), X222[0, :, :, args.depth // 2])


        generator.save("%s/%s-it%05d.h5" % (itr_out_dir, args.expName, epoch), \
                       include_optimizer=False)


    sys.stdout.flush()
