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
from chainer.datasets import (TupleDataset, TransformDataset)
from chainer.links.model.vision import resnet
from chainercv import transforms

#PATH関連
# このファイルの絶対パス
FILE_PATH = path.dirname(path.abspath(__file__))
# deep learningディレクトリのrootパス
ROOT_PATH = path.normpath(path.join(FILE_PATH, '../'))

class WideResNet(chainer.Chain):
    '''
    wide resnet model
    '''
    def __init__(self, widerate=4, num_class=10, blocks=(3, 4, 6)):
        super(WideResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16 * widerate, ksize=3, stride=1)
            self.bn1 = L.BatchNormalization(16 * widerate)
            self.block2 = resnet.BuildingBlock(blocks[0], 16 * widerate, 16 * widerate, 32 * widerate, stride=2)
            self.block3 = resnet.BuildingBlock(blocks[1], 32 * widerate, 32 * widerate, 64 * widerate, stride=2)
            self.block4 = resnet.BuildingBlock(blocks[2], 64 * widerate, 64 * widerate, 128 * widerate, stride=2)
            self.fc5 = L.Linear(128 * widerate, num_class)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        nbatch, ch, row, col = h.data.shape
        h = F.average_pooling_2d(h, (row, col), stride=1)
        h = F.reshape(h, (nbatch, ch))
        h = self.fc5(h)
        return h

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

def load_dataset():
    '''
    loading dataset
    '''
    # loaading dataet from npz file
    data = np.load(path.join(ROOT_PATH, 'data/cifar10.npz'))
    train_x, train_y = data['train_x'], data['train_y']
    test_x, test_y = data['test_x'], data['test_y']

    # reshaping and casting to flaot
    train_x = train_x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_x = test_x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    # calc pixel per
    ppm = train_x.mean(axis=0)
    train = TupleDataset(train_x - ppm, train_y)
    test = TupleDataset(test_x - ppm, test_y)

    # data augmentation
    train = TransformDataset(train, transform)

    return train, test


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
    print('# loading CIFAR-10')
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
