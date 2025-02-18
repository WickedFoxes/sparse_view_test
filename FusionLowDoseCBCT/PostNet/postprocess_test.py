import time
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from skimage import measure
import scipy.io as io
import imageio
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--fdk_dir', type=str, default="../walnut19_div2_interpolation_fkd", help=' : Please set the dir')
# parser.add_argument('--post_dir', type=str, default="../walnut19_div2_interpolation_fkd_post", help=' : Please set the dir')
parser.add_argument('--post_dir', type=str, default="../walnut19_div2_interpolation_fkd_post", help=' : Please set the dir')

args = parser.parse_args()
fdkDir = args.fdk_dir
postDir = args.post_dir

print(args)

tf.reset_default_graph()
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
gen_in = tf.placeholder(shape=[1, 448, 448, 9, 1], dtype=tf.float32,
                        name='generated_image')
Gz = generator(gen_in)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = initialize(sess)
    initial_step = global_step.eval()
    interval = 0
    for cc in range(1):
        #idx=cc+19
        #print(idx)
        if not os.path.exists(postDir):
            os.makedirs(postDir)

        recon = np.zeros((448, 448, 200), dtype=np.float32)
        for cc in range(0, 200):
            print(cc+1)
            # a=imageio.imread(os.path.join(fdkDir,f"fdk_{(cc+1):06d}.png")).astype(float)
            # rand a
            a = np.random.rand(448, 448)
            recon[:,:,cc] = (a-a.min())/(a.max()-a.min())

        print('Testing')
        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(1)
        batch[:,:,0,0] = recon[:,:,0]
        batch[:,:,1,0] = recon[:,:,0]
        batch[:,:,2,0] = recon[:,:,0]
        batch[:,:,3,0] = recon[:,:,0]
        batch[:,:,4,0] = recon[:,:,0]
        batch[:,:,5,0] = recon[:,:,1]
        batch[:,:,6,0] = recon[:,:,2]
        batch[:,:,7,0] = recon[:,:,3]
        batch[:,:,8,0] = recon[:,:,4]
        image = np.expand_dims(batch,axis=0)
        print(image.shape, "image shape 1")
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(1):06d}.png"), img)

        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(2)
        batch[:,:,0,0] = recon[:,:,0]
        batch[:,:,1,0] = recon[:,:,0]
        batch[:,:,2,0] = recon[:,:,0]
        batch[:,:,3,0] = recon[:,:,0]
        batch[:,:,4,0] = recon[:,:,1]
        batch[:,:,5,0] = recon[:,:,2]
        batch[:,:,6,0] = recon[:,:,3]
        batch[:,:,7,0] = recon[:,:,4]
        batch[:,:,8,0] = recon[:,:,5]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(2):06d}.png"), img)


        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(3)
        batch[:,:,0,0] = recon[:,:,0]
        batch[:,:,1,0] = recon[:,:,0]
        batch[:,:,2,0] = recon[:,:,0]
        batch[:,:,3,0] = recon[:,:,1]
        batch[:,:,4,0] = recon[:,:,2]
        batch[:,:,5,0] = recon[:,:,3]
        batch[:,:,6,0] = recon[:,:,4]
        batch[:,:,7,0] = recon[:,:,5]
        batch[:,:,8,0] = recon[:,:,6]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(3):06d}.png"), img)



        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(4)
        batch[:,:,0,0] = recon[:,:,0]
        batch[:,:,1,0] = recon[:,:,0]
        batch[:,:,2,0] = recon[:,:,1]
        batch[:,:,3,0] = recon[:,:,2]
        batch[:,:,4,0] = recon[:,:,3]
        batch[:,:,5,0] = recon[:,:,4]
        batch[:,:,6,0] = recon[:,:,5]
        batch[:,:,7,0] = recon[:,:,6]
        batch[:,:,8,0] = recon[:,:,7]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(4):06d}.png"), img)


        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        for i in tqdm(range(192)):
            print(i+5)
            batch[:,:,0,0] = recon[:,:,i]
            batch[:,:,1,0] = recon[:,:,i+1]
            batch[:,:,2,0] = recon[:,:,i+2]
            batch[:,:,3,0] = recon[:,:,i+3]
            batch[:,:,4,0] = recon[:,:,i+4]
            batch[:,:,5,0] = recon[:,:,i+5]
            batch[:,:,6,0] = recon[:,:,i+6]
            batch[:,:,7,0] = recon[:,:,i+7]
            batch[:,:,8,0] = recon[:,:,i+8]
            image = np.expand_dims(batch,axis=0)
            print(image.shape)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            img=255.0*(img-img.min())/(img.max()-img.min())
            img=img.astype(np.uint8)
            imageio.imsave(os.path.join(postDir, f"fdk_{(i+5):06d}.png"), img)



        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(197)
        batch[:,:,0,0] = recon[:,:,192]
        batch[:,:,1,0] = recon[:,:,193]
        batch[:,:,2,0] = recon[:,:,194]
        batch[:,:,3,0] = recon[:,:,195]
        batch[:,:,4,0] = recon[:,:,196]
        batch[:,:,5,0] = recon[:,:,197]
        batch[:,:,6,0] = recon[:,:,198]
        batch[:,:,7,0] = recon[:,:,199]
        batch[:,:,8,0] = recon[:,:,199]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(197):06d}.png"), img)


        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(198)
        batch[:,:,0,0] = recon[:,:,193]
        batch[:,:,1,0] = recon[:,:,194]
        batch[:,:,2,0] = recon[:,:,195]
        batch[:,:,3,0] = recon[:,:,196]
        batch[:,:,4,0] = recon[:,:,197]
        batch[:,:,5,0] = recon[:,:,198]
        batch[:,:,6,0] = recon[:,:,199]
        batch[:,:,7,0] = recon[:,:,199]
        batch[:,:,8,0] = recon[:,:,199]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(198):06d}.png"), img)



        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(199)
        batch[:,:,0,0] = recon[:,:,194]
        batch[:,:,1,0] = recon[:,:,195]
        batch[:,:,2,0] = recon[:,:,196]
        batch[:,:,3,0] = recon[:,:,197]
        batch[:,:,4,0] = recon[:,:,198]
        batch[:,:,5,0] = recon[:,:,199]
        batch[:,:,6,0] = recon[:,:,199]
        batch[:,:,7,0] = recon[:,:,199]
        batch[:,:,8,0] = recon[:,:,199]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(199):06d}.png"), img)



        batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
        print(200)
        batch[:,:,0,0] = recon[:,:,195]
        batch[:,:,1,0] = recon[:,:,196]
        batch[:,:,2,0] = recon[:,:,197]
        batch[:,:,3,0] = recon[:,:,198]
        batch[:,:,4,0] = recon[:,:,199]
        batch[:,:,5,0] = recon[:,:,199]
        batch[:,:,6,0] = recon[:,:,199]
        batch[:,:,7,0] = recon[:,:,199]
        batch[:,:,8,0] = recon[:,:,199]
        image = np.expand_dims(batch,axis=0)
        image_recon = sess.run(Gz, feed_dict={gen_in: image})
        image_recon = np.resize(image_recon,[448,448,9])
        img = image_recon[:,:,4]
        img[img < 0.0]=0.0
        img[img > 1.0]=1.0
        img=255.0*(img-img.min())/(img.max()-img.min())
        img=img.astype(np.uint8)
        imageio.imsave(os.path.join(postDir, f"fdk_{(200):06d}.png"), img)