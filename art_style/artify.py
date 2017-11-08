import sys
import time
import argparse
import numpy as np
from skimage import io
import chainer
import chainer.links as L
import chainer.functions as F
import cupy as cp
from tqdm import tqdm
import copy

MAX_ITERS = 5000

class GenImage(chainer.Chain):
    def __init__(self, input_img):
        super(GenImage, self).__init__()
        with self.init_scope():
            init_img = np.random.rand(*input_img.shape).astype(np.float32)
            self.b = chainer.Parameter(init_img)

class VGG16(chainer.Chain):
    def __init__(self):
        super(VGG16, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            # self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            # self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            # self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
            # self.fc6 = L.Linear(512 * 7 * 7, 4096)
            # self.fc7 = L.Linear(4096, 4096)
            # self.fc8 = L.Linear(4096, 1000)

    def __call__(self, x):
        # 1 Layer
        h  = F.relu(self.conv1_1(x))
        h1 = F.relu(self.conv1_2(h))
        # 2 Layer
        h  = F.max_pooling_2d(h1, ksize=2)
        h  = F.relu(self.conv2_1(h))
        h2 = F.relu(self.conv2_2(h))
        # 3 Layer
        h  = F.max_pooling_2d(h2, ksize=2)
        h  = F.relu(self.conv3_1(h))
        h  = F.relu(self.conv3_2(h))
        h3 = F.relu(self.conv3_3(h))
        # 4 Layer
        h  = F.max_pooling_2d(h3, ksize=2)
        h  = F.relu(self.conv4_1(h))
        h  = F.relu(self.conv4_2(h))
        h4 = F.relu(self.conv4_3(h))
        return h1, h2, h3, h4

def get_matrix(y):
    '''
    入力map:yは1, ch, H, W
    '''
    _, ch, h, w = y.shape
    buf = F.reshape(y, (ch, h * w))
    matrix = F.matmul(buf, buf, transb=True) / np.float32(ch * h * w)
    return matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', '-i', type=int, default=2000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', '-l', type=float, default=4.0,
                        help='adam lr')
    parser.add_argument('--ratio', '-r', type=float, default=0.005,
                        help='ratio of (origin image / style image)')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--image', '-i', type=str, default='img_01',
                        help='input image')
    parser.add_argument('--style', '-s', type=str, default='style_02',
                        help='style image')
    args = parser.parse_args()

    # 入力画像読み込み
    input_img = cp.array(io.imread('./img/{}.bmp'.format(args.style)), dtype=np.float32)
    input_img = input_img.reshape(1, *input_img.shape).transpose(0, 3, 1, 2)
    # input_img = (input_img - 127) #/ 255.0
    # スタイル画像読み込み
    style_img = cp.array(io.imread('./img/{}.bmp'.format(args.style)), dtype=np.float32)
    style_img = style_img.reshape(1, *style_img.shape).transpose(0, 3, 1, 2)
    # style_img = (style_img - 127) #/ 255.0

    # モデル読み込み
    model = VGG16()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    chainer.serializers.load_npz('./VGG16.npz', model)

    # 学習初期化
    # 入力画像の隠れ層とスタイル画像のスタイル行列
    input_map = model(input_img)
    style_mat = [get_matrix(y) for y in model(style_img)]

    # 生成画像初期化
    # gen_img = GenImage(input_img)
    gen_img = chainer.links.Bias(axis=1, shape=input_img.shape)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen_img.to_gpu()

    # 最適化計算 初期化
    optimizer = chainer.optimizers.Adam(args.lr)
    optimizer.setup(gen_img)

    # マップ,スタイルの各層の学習係数
    alpha = [0, 0, 1, 1]
    beta = [1, 1, 1, 1]
    # 学習ループ
    for itr in tqdm(range(args.iter)):
        gen_map = model(gen_img.b)

        # 各層に対して誤差計算
        loss = None
        for i, m in enumerate(gen_map):
            mat = get_matrix(m)

            # マップの誤差
            loss1 = args.ratio  * alpha[i] * F.mean_squared_error(m, input_map[i])
            #スタイル行列の誤差
            loss2 = beta[i] * F.mean_squared_error(mat, style_mat[i]) / np.float32(len(gen_map))
            hasattr(loss1, 'back')
            if loss is None:
                loss = loss1 + loss2
            else:
                loss += loss1 + loss2

        gen_img.cleargrads()
        loss.backward()
        optimizer.update()
        # print(itr, loss)
        sys.stdout.write('\rIter:{:>4}  Loss:{:.4f}\n'.format(itr, float(loss.data)))

        # 途中画像を保存
        if itr % (args.iter // 50) == 0:
            _, ch, h, w = input_img.shape
            img = chainer.cuda.to_cpu(gen_img.b.data)
            print(img.min(), img.max(), img.mean())
            img2 = (img - img.min()) / (img.max() - img.min()) * 255
            img[img < 0] = 0
            img[img > 255] = 255
            print(img.min(), img.max(), img.mean())
            print(img2.min(), img2.max(), img2.mean())
            img = (img.reshape(ch, h, w).transpose(1,2,0)).astype(np.uint8)
            img2 = (img2.reshape(ch, h, w).transpose(1,2,0)).astype(np.uint8)
            io.imsave('./results/img1_{}_iter_{}.bmp'.format(args.style, itr), img)
            io.imsave('./results/img2_{}_iter_{}.bmp'.format(args.style, itr), img2)

