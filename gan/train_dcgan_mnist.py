'''
    Distillation from donor model
'''
import os
from os import path
import argparse
import random
import platform

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.links.model.vision import resnet
from chainercv import transforms

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

class Generator(chainer.Chain):
    def __init__(self, ch=512, bw=4, wscale=0.02):
        self.ch = ch
        self.bw = bw
        self.wscale = wscale
        super(Generator, self).__init__()
        with self.init_scope():
            init_w = chainer.initializers.Normal(wscale)
            self.fc = L.Linear(100, bw * bw * ch, initialW=init_w)
            self.dconv1 = L.Deconvolution2D(None, ch // 2, ksize=4, stride=2, pad=1, initialW=init_w)
            self.dconv2 = L.Deconvolution2D(None, ch // 4, ksize=4, stride=2, pad=1, initialW=init_w)
            self.dconv3 = L.Deconvolution2D(None, ch // 8, ksize=4, stride=2, pad=1, initialW=init_w)
            self.dconv4 = L.Deconvolution2D(None, 1, ksize=4, stride=2, pad=1, initialW=init_w)
            self.bn0 = L.BatchNormalization(bw * bw * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.fc(z))), (len(z), self.ch, self.bw, self.bw))
        h = F.relu(self.bn1(self.dconv1(h)))
        h = F.relu(self.bn2(self.dconv2(h)))
        h = F.relu(self.bn3(self.dconv3(h)))
        x = F.tanh(self.dconv4(h))
        return x

class Discriminator(chainer.Chain):
    def __init__(self, ch=512, bw=4, wscale=0.02):
        self.ch = ch
        self.bw = bw
        self.wscale = wscale
        super(Discriminator, self).__init__()
        with self.init_scope():
            init_w = chainer.initializers.Normal(wscale)
            self.conv1_0 = L.Convolution2D(None, ch // 8, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv1_1 = L.Convolution2D(None, ch // 4, ksize=4, stride=2, pad=1, initialW=init_w)
            self.conv2_0 = L.Convolution2D(None, ch // 4, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv2_1 = L.Convolution2D(None, ch // 2, ksize=4, stride=2, pad=1, initialW=init_w)
            self.conv3_0 = L.Convolution2D(None, ch // 2, ksize=3, stride=1, pad=1, initialW=init_w)
            self.conv3_1 = L.Convolution2D(None, ch // 1, ksize=4, stride=2, pad=1, initialW=init_w)
            self.conv4 = L.Convolution2D(None, ch // 1, ksize=3, stride=1, pad=1, initialW=init_w)
            self.l5 = L.Linear(bw * bw * ch, 1, initialW=init_w)
            self.bn1_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn3_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn4_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        h = F.leaky_relu(self.conv1_0(x))
        h = F.leaky_relu(self.bn1_1(self.conv1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.conv2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.conv2_1(h)))
        h = F.leaky_relu(self.bn3_0(self.conv3_0(h)))
        h = F.leaky_relu(self.bn3_1(self.conv3_1(h)))
        h = F.leaky_relu(self.bn4_0(self.conv4(h)))
        h = self.l5(h)
        return h

class DCGANUpdater(chainer.training.StandardUpdater):
    def __init__():
        pass

    def loss_dis(self, dis)

def main():
    '''
    main function, start point
    '''
    # 引数関連
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    # save didrectory
    outdir = path.join(FILE_PATH, 'results/DCGAN_mnist')
    if not path.exists(outdir):
        os.makedirs(outdir)
    with open(path.join(outdir, 'arg_param.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k,v))

    # print param
    print('# GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# Epoch: {}'.format(args.epoch))
    print('# Dataset: CIFAR-10')
    print('# Learning-rate :{}'.format(args.learnrate))
    print('# out directory :{}'.format(outdir))
    print('')

    #loading dataset
    train, _ = chainer.datasets.get_mnist()

    # prepare model
    gen = Generator()
    dis = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # setup optimizer
    opt_gen = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_gen.setup(gen)
    opt_gen.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    # setup optimizer
    opt_dis = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    opt_dis.setup(dis)
    opt_dis.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')

    # setup iter
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    trainer.run()

if __name__ == '__main__':
    main()
