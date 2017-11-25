import tensorflow as tf
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import sys
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import dcgan_var2 as dcgan
#import dcgan


if __name__ == "__main__":
    #dp = dcgan.DataProcess("../dataset/102flowers/jpg/")
    np.random.seed(1)
    g = dcgan.Generator()
    d = dcgan.Discriminator(g)
    gan = dcgan.GANModel(generator=g, discriminator=d, datasrc=None)
    
    #imgs = gan.infer_gen(model_path=sys.argv[1], n_img = 1)
    #img = np.reshape(imgs, (64, 64, 3))
    #plt.imshow(img/256.0)
    

    imgs = gan.infer_gen(model_path=sys.argv[1], n_img = 100)

    #print(np.sum(np.abs(imgs[1,:,:,:] - imgs[2,:,:,:])))

    img = []
    for i in range(10):
        tmp = []
        for j in range(10):
            tmp.append(imgs[i*10 + j, :, :, :])
        img.append(np.concatenate(tmp, axis = 0))
    img = np.concatenate(img, axis = 1)
    print(img.shape)
    plt.imshow(img/256.0)
    plt.show()
