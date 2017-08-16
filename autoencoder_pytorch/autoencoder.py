
import sys
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class DataIterator(object):
    """
        Suppose data is a numpy array
    """
    def __init__(self, data_in, batch_size = 10):
        self.data_in = copy.deepcopy(data_in)
        np.random.shuffle(self.data_in)
        self.batch_size = batch_size
        self.datalen = len(self.data_in)

    def __iter__(self):
        self.current = 0
        return self

    def next(self):
        return self.__next__() 
        
    def __next__(self):
        if self.current >= self.datalen: raise StopIteration
        ret = self.data_in[self.current: self.current+self.batch_size]
        self.current += self.batch_size
        return ret

class AutoEncoder(nn.Module):
    def __init__(self, in_size, out_size, batch_num = 10, epoch_num = 10):
        """
            in_size: Data Input Dimension
            out_size: Data Output Dimension
            batch_num: Batch size of Input
            epoch_num: Training Epoches
        """
        super(AutoEncoder, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.batch_num = batch_num
        self.epoch_num = epoch_num

        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, in_size)
        
    def forward(self, data_in):
        data_out = F.sigmoid(self.linear1(data_in))
        data_out = F.sigmoid(self.linear2(data_out))
        return data_out

    def fit_transform(self, data_in):
        criterion = nn.MSELoss()
        # for gpu applications
        self.cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        for epoch in range(self.epoch_num):
            for data in DataIterator(data_in, self.batch_num):
                data = torch.from_numpy(data)
                data = data.float()
                data = data.cuda()
                data = Variable(data)

                optimizer.zero_grad()

                # Sparse Autoencoders, L2-norm, this is proved to be have nearly the same effect as original data
                # L1-norm will produce worse models
                # tanh will produce worse models, may be this indicates the figure need normalization
                # loss = criterion(self(data), data) + 0.0001*(F.sigmoid(self.linear1(data))**2).sum()
                # another loss function will be CAE (contractive autoencoder)

                hidden = F.sigmoid(self.linear1(data))
                #gradients = torch.autograd.grad(inputs = data, outputs = hidden, retain_graph = True, create_graph = True)
                
                # PyTorch evaluating Jacobian Matrix is extremely complicated!
                # Thanks to this blog:
                # https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/

                hw = hidden*(1.0 - hidden)
                # CAE is still not enough, only give ~81%, compared to the original data ~86%
                loss = criterion(self(data), data) + 0.01*(hw * (self.linear1.weight**2).sum(dim = 1)).sum()

                loss.backward()
                optimizer.step()
                sys.stdout.write("In epoch %d, total loss %.6f\r" %(epoch, loss.data.cpu().numpy()))
            #print("")
        print("")

        return F.sigmoid(self.linear1(Variable(torch.from_numpy(data_in).cuda()))).data.cpu().numpy()
        #return self(Variable(torch.from_numpy(data_in).cuda())).data.cpu().numpy()

    def transform(self, data_in):
        return F.sigmoid(self.linear1(Variable(torch.from_numpy(data_in).cuda()))).data.cpu().numpy()

class StackedAutoEncoder(object):
    def __init__(self, in_size, out_size, data_in):
        self.in_size = in_size
        self.out_size = out_size
        self.data_in = data_in
        self.netlist = []

    def fit_transform(self, data_in):
        in_size = self.in_size
        current_data = data_in

        for idx, sz in enumerate(self.out_size):
            print("Starting training AutoEncoder layer %d" %idx)
            #print(current_data.shape)
            #print(self.out_size)
            ae = AutoEncoder(in_size, sz, batch_num = 100, epoch_num = 100)
            current_data = ae.fit_transform(current_data)
            self.netlist.append(ae)
            in_size = sz

        return current_data

    def transform(self, data_in):
        tmp = data_in
        for net in self.netlist:
            tmp = net.transform(tmp)
        return tmp

if __name__ == "__main__":
    print torch.__version__
    sys.path.append("../") 
    import DataReader
    tdata = DataReader.ImageReader("../dataset/train-images-idx3-ubyte.gz").to_tensor()
    ldata = DataReader.LabelReader("../dataset/train-labels-idx1-ubyte.gz").to_tensor()
    print tdata.shape
    print ldata.shape

    import sklearn
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    saen = StackedAutoEncoder(28*28, (400, ), tdata)
    ret = saen.fit_transform(tdata)

    #import matplotlib.pyplot as plt
    #graphs, axes = plt.subplots(nrows=5, ncols=5)
    #axes = axes.flatten()

    #for t in range(25):
    #    axes[t].imshow(ret[t,:].reshape((28,28)),  cmap='gray')
    #    graphs.tight_layout()
    #plt.show()
    
    print("Training Model1 ...")
    model1 = MultiOutputClassifier(sklearn.ensemble.RandomForestClassifier(10))
    model1.fit(saen.transform(tdata), ldata)

    print("Training Model2 ...")
    model2 = MultiOutputClassifier(sklearn.ensemble.RandomForestClassifier(10))
    model2.fit(tdata, ldata)
    
    ttest = DataReader.ImageReader("../dataset/t10k-images-idx3-ubyte.gz").to_tensor()
    ltest = DataReader.LabelReader("../dataset/t10k-labels-idx1-ubyte.gz").to_tensor()
    
    pred1 = model1.predict(saen.transform(ttest))
    pred2 = model2.predict(ttest)

    accu1 = accuracy_score(pred1, ltest)
    accu2 = accuracy_score(pred2, ltest)

    print("Model 1 Accuracy (with SEAN): %.3f, Model 2 Accuracy: %.3f" %(accu1, accu2))
