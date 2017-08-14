#! /usr/bin/env python

"""
    DC-GAN implemention using pytorch
    DataSet cifar-10
    A Wasserstein version
    @author Victor (Xiao-Jie) Zhang
    @date 2017/08/10
"""

#import collections
import cPickle as pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from torch.nn.functional import leaky_relu, tanh, sigmoid
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys

class DataReader(Dataset):
    """
        A simple Data Reader for cifar-10 image set
    """
    def __init__(self, datasrc, transform = None):
        self.data = []
        if not isinstance(datasrc, str):
            for fn in datasrc:
                with open(fn, "r") as fin:
                    self.data.append(pickle.load(fin)["data"])
            self.data = np.vstack(self.data)
            #print self.data.shape
            #sys.exit()
        else:
            with open(datasrc, "r") as fin:
                self.data = pickle.load(fin)["data"]
        self.transform = transform
        self.length = len(self.data)
        print("Image number: %d" %self.length)
        #print self.data['label']
        #self._draw()

    def __getitem__(self, idx):
        if not self.transform:
            return self.data[idx]
        else:
            return self.transform(self.data[idx].reshape(3, 32, 32).transpose((1,2,0)))

    def __len__(self):
        return self.length
                
    def _draw(self):
        # Draw a series of images, just for testing
        data = []
        cnt = 0
        for i in range(10):
            tmp = []
            for j in range(10):
                tmp.append(self.data[cnt].reshape(3, 32, 32).transpose((1,2,0)))
                cnt += 1
            data.append(np.hstack(tmp))
        data = np.vstack(data)
        plt.imshow(data)
        plt.show()

class DCGenerator(torch.nn.Module):
    def __init__(self, prior_size):
        super(DCGenerator, self).__init__()
        self.prior_size = prior_size
        self.linear1 = nn.Linear(prior_size, 4*4*512)
        # 4x4 --> 8x8
        self.deconv1 = nn.ConvTranspose2d(512, 256, (5,5))
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(256)

        # 8x8 --> 16x16, stride 2
        self.deconv2 = nn.ConvTranspose2d(256, 128, (5,5), stride = (2,2), padding = (2,2), output_padding = (1,1))
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(128)

        # 16x16 --> 32x32, stride
        self.deconv3 = nn.ConvTranspose2d(128, 3, (5,5), stride = (2,2), padding = (2,2), output_padding = (1,1))

    def forward(self, prior):
        prior = prior.cuda()
        fc_layer = leaky_relu(self.linear1(prior).view(-1, 512, 4, 4), negative_slope = 0.2)
        deconv_layer1 = self.bn1(leaky_relu(self.deconv1(fc_layer), negative_slope = 0.2))
        deconv_layer2 = self.bn2(leaky_relu(self.deconv2(deconv_layer1), negative_slope = 0.2))
        deconv_layer3 = tanh(self.deconv3(deconv_layer2))
        return deconv_layer3

class DCDiscriminator(torch.nn.Module):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, (5,5), padding = (2,2), stride = (2,2))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, (5,5), padding = (2,2), stride = (2,2))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, (5,5), padding = (2,2), stride = (2,2))
        self.linear1 = nn.Linear(4*4*512, 1)

    def forward(self, image):
        image = image.cuda()
        conv_layer1 = self.bn1(leaky_relu(self.conv1(image), negative_slope = 0.2))
        conv_layer2 = self.bn2(leaky_relu(self.conv2(conv_layer1), negative_slope = 0.2))
        conv_layer3 = leaky_relu(self.conv3(conv_layer2), negative_slope = 0.2)
        fc_layer1 = self.linear1(conv_layer3.view(-1, 4*4*512))
        return fc_layer1

if __name__ == "__main__":
    BATCH_SIZE = 100
    import glob
    d = DataReader(glob.glob("../dataset/cifar-10-batches-py/data_batch_*"), transform = \
        transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2.0*x-1.0)]))
    #print d[0]
    dcg = DCGenerator(100)
    dcd = DCDiscriminator()

    #print(dcd(img))
    dcg.cuda()
    dcd.cuda()
   
    # define optimizer for Generator and Discriminitor
    optimizer_g = torch.optim.RMSprop( dcg.parameters(), lr = 5e-4)
    optimizer_d = torch.optim.RMSprop( dcd.parameters(), lr = 5e-4)
    
    for epoch in range(10):
        dataloader = DataLoader(d, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory =True)
        for batch in dataloader:

            gen = dcg(Variable(torch.randn(BATCH_SIZE,100)*2.0e-2))
            dis_real = dcd(Variable(batch))
            dis_fake = dcd(gen)

            optimizer_d.zero_grad()
            # Loss for Discriminators
            #loss2 = nn.functional.binary_cross_entropy(dis_fake, Variable(torch.zeros(dis_fake.size()).cuda()) )
            #loss3 = nn.functional.binary_cross_entropy(dis_real, Variable(torch.ones(dis_fake.size() ).cuda()) )
            loss_d = - dis_real.mean() + dis_fake.mean()
            loss_d.backward(retain_variables=True)
            optimizer_d.step()

            for param in dcd.parameters():
                param.data.clamp_(-0.01, 0.01)

            #print(type(dis_real), type(dis_fake))
            optimizer_g.zero_grad()
            # Loss for Generators
            # loss1 = nn.functional.binary_cross_entropy(dis_fake, Variable(torch.ones(dis_fake.size()).cuda() ) )
            #loss1 = dis_fake.mean()
            loss_g = -dis_fake.mean()
            loss_g.backward()
            optimizer_g.step()

            sys.stdout.write("In epoch %d, Generator Loss %.3f, Discriminator Loss %.3f\r" \
                               %(epoch, loss_g.cpu().data.numpy()[0], loss_d.cpu().data.numpy()[0]))

    print("")
    print("Finish Training.... Now Dispaly Several GAN Images:")
    gen = dcg(Variable(torch.randn(100,100)*2.0e-2))
    gen = (gen + 1)/2
    gen = (gen*256.0).cpu().data.numpy().astype(np.uint8)

    data = []
    cnt = 0
    for i in range(10):
        tmp = []
        for j in range(10):
            tmp.append(gen[cnt].reshape(3, 32, 32).transpose((1,2,0)))
            cnt += 1
        data.append(np.hstack(tmp))
    data = np.vstack(data)
    plt.imshow(data)
    plt.show()

