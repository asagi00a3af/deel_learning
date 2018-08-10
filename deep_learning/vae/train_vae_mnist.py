"""
train vae, mnist
chainer の example の写経
"""
import os
from os import path
import argparse
import platform
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence
from chainer.training import extensions
from chainer import training
import matplotlib.pyplot as plt
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use('Agg')

class VAE(chainer.Chain):
    def __init__(self, n_in, n_latent, n_hidden):
        super(VAE, self).__init__()
        with self.init_scope():
            # enc
            self.le1 = L.Linear(n_in, n_hidden)
            self.le2_mu = L.Linear(n_hidden, n_latent)
            self.le2_ln_var = L.Linear(n_hidden, n_latent)
            #dec
            self.ld1 = L.Linear(n_latent, n_hidden)
            self.ld2 = L.Linear(n_hidden, n_in)

    def __call__(self, x, sigmoid=True):
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h = F.tanh(self.le1(x))
        mu = self.le2_mu(h)
        ln_var = self.le2_ln_var(h)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h = F.tanh(self.ld1(z))
        h = self.ld2(h)
        if sigmoid:
            return F.sigmoid(h)
        else:
            return h

    def get_loss_func(self, C=1.0, k=1):
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            rec_loss = 0
            for l in range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + C * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report({'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf

def save_reconstructed_images(x, x1, filename):
    fig, ax = plt.subplots(1, 2, figsize=(18, 9), dpi=100)
    for ai, x in zip(ax.flatten(), (x, x1)):
        x = x.reshape(10, 10, 28, 28).transpose(0,2,1,3).reshape(280, 280)
        ai.imshow(x)
    fig.savefig(filename)

def save_sampled_images(x, filename):
    x = x.reshape(10, 10, 28, 28).transpose(0,2,1,3).reshape(280, 280)
    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(x)
    fig.savefig(filename)
def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--coef', '-c', type=float, default=1.0,
                        help='')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    outdir = path.join('results', 'VAE', 'VAE_dimz{}_coef{}'.format(args.dimz, args.coef))
    print("# result dir : {}".format(outdir))
    if not path.exists(outdir):
        os.makedirs(outdir)

    model = VAE(784, args.dimz, 500)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    if args.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,repeat=False, shuffle=False)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=model.get_loss_func(C=args.coef))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu, eval_func=model.get_loss_func(C=args.coef, k=10)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/rec_loss', 'validation/main/rec_loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    model.to_cpu()
    #draw reconstructred image
    x = np.array(train[:100])
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x).data
    save_reconstructed_images(x, x1, os.path.join(outdir, 'train_reconstructed'))

    x = np.array(test[:100])
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x).data
    save_reconstructed_images(x, x1, os.path.join(outdir, 'test_reconstructed'))

    # draw images from randomly sampled z
    z1, z2 = np.random.normal(0, 1, (2, args.dimz))
    z = z1 + np.kron(np.linspace(0, 1, 100).reshape(100, 1), (z2 - z1))
    x = model.decode(z.astype(np.float32)).data
    save_sampled_images(x, os.path.join(outdir, 'sampled'))

if __name__ == '__main__':
    main()
