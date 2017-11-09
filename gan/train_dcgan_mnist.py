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
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(100, 4 * 4 * 1024)
            self.bn0 = L.BatchNormalization(4 * 4 * 1024)
            self.dconv1 = L.Deconvolution2D(None, 512, ksize=4, stride=2, pad=1)
            self.bn1 = L.BatchNormalization(512)
            self.dconv2 = L.Deconvolution2D(None, 256, ksize=4, stride=2, pad=1)
            self.bn2 = L.BatchNormalization(256)
            self.dconv3 = L.Deconvolution2D(None, 128, ksize=4, stride=2, pad=1)
            self.bn3 = L.BatchNormalization(128)
            self.dconv4 = L.Deconvolution2D(None, 1, ksize=4, stride=2, pad=1)

    def __call__(self, z):
        h = F.reshape(F.relu(self.fc(z)), (z.shape[0], 1024, 4, 4))
        h = F.relu(self.bn1(self.dconv1(h)))
        h = F.relu(self.bn2(self.dconv2(h)))
        h = F.relu(self.bn3(self.dconv3(h)))
        x = F.tanh(self.dconv4(h))
        return x

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=4, st)

    def __call__(self, z):


def transform(data):
    '''
    data aug function
    '''
    img, label = data
    # random crop
    img = transforms.random_crop(img, (24, 24))
    # random flip
    img = transforms.random_flip(img, y_random=True, x_random=True)
    # random rot
    transforms.random_rotate(img)
    if random.choice([True, False]):
        transforms.pca_lighting(img, 0.1)
    return img, label

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
    parser.add_argument('--widerate', type=int, default=1, help='WideResnet wide parameter')
    parser.add_argument('--blocks', nargs='+', type=int,
                        default=[1, 1, 1], help='WideResnet blocks parameter')
    args = parser.parse_args()

    # save didrectory
    outdir = path.join(FILE_PATH, 'results/cifar10_wideresnet_{}_{}_lr_{}_lr_shift'.format(
        args.widerate, ''.join(map(str, args.blocks)), args.learnrate))
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
    print('# Wide Resnet widerate :{}'.format(args.widerate))
    print('# Wide Resnet blocks :{}'.format(args.blocks))
    print('# out directory :{}'.format(outdir))
    print('')

    #loading dataset
    train, test = load_dataset()

    # prepare model
    model = L.Classifier(WideResNet(num_class=10, widerate=args.widerate, blocks=args.blocks))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # setup optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate)
    optimizer.setup(model)

    # setup iter
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    # eval test data
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    # dump loss graph
    trainer.extend(extensions.dump_graph('main/loss'))
    # lr shift
    trainer.extend(extensions.ExponentialShift("lr", 0.1), trigger=(100, 'epoch'))
    # save snapshot
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_snapshot_{.updater.epoch}'), trigger=(10, 'epoch'))
    # log report
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
    #  plot loss graph
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                'epoch', file_name='loss.png'))
    # plot acc graph
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'))
    # print info
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time']))
    # print progbar
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
