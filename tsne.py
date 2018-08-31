# _*_ coding: utf-8 _*_
import sys
import pickle
import argparse
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=32,
                          help='# of samples to compute embeddings on. Becomes slow if very high.')
    parser.add_argument('--x_dir', type=str, default='D:/Desktop/hanchun.shen/Desktop/feats_from_yanhong/06272/512/signatures.npy',
                        help='Directory where plots are saved')
    parser.add_argument('--y_dir', type=str, default='D:/Desktop/hanchun.shen/Desktop/feats_from_yanhong/06272/512/gallery.npy',
                        help='Directory where plots are saved')
    parser.add_argument('--num_dimensions', type=int, default=3,
                          help='# of tsne dimensions. Can be 2 or 3.')
    parser.add_argument('--shuffle', type=bool, default=False,
                          help='Whether to shuffle the data before embedding.')
    parser.add_argument('--compute_embeddings', type=bool, default=True,
                          help='Whether to compute embeddings. Do this once per sample size.')
    parser.add_argument('--with_images', type=bool, default=False,
                          help='Whether to overlay images on data points. Only works with 2D plots.')
    parser.add_argument('--random_seed', type=int, default=42,
                          help='Seed to ensure reproducibility')
    parser.add_argument('--data_dir', type=str, default='./sources/',
                          help='Directory where data is stored')
    parser.add_argument('--plot_dir', type=str, default=None,
                          help='Directory where plots are saved')
    

    return parser.parse_args()

class tsne(object):
    def __init__(self):
        self.__x_sample = None
        self.__y_sample = None
        self.__num_classes = None
        self._args = get_args()
        self.__labels = None

    def __data_loader(self):
        x_train = np.load(self._args.x_dir)
        y_train = np.load(self._args.y_dir)
        print '-' * 70
        print 'source features shape: ', x_train.shape
        print 'source labels shape: ', y_train.shape
        return x_train, y_train

    def _plot2D(self, embeddings):
        args = get_args()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors = cm.Spectral(np.linspace(0, 1, self.__num_classes))

        xx = embeddings[:, 0]
        yy = embeddings[:, 1]

        if args.with_images == True:
            for i, d in enumerate(zip(xx, yy)):
                x, y = d
                im = OffsetImage(self.__x_sample[i], zoom=0.1, cmap='gray')
                ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                ax.add_artist(ab)
            ax.update_datalim(np.column_stack([xx, yy]))
            ax.autoscale()

        for i in range(self.__num_classes):
            ax.scatter(xx[self.__y_sample==i], yy[self.__y_sample==i], color=colors[i], label=self.__labels[i], s=60)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        if self._args.plot_dir:
            plt.savefig(self._args.plot_dir, dpi=900)
            print 'fig\'s been saved!'
        plt.show()

    def _plot3D(self, embeddings):
        if self._args.with_images:
            sys.exit("Cannot plot images with 3D plots.")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = cm.Spectral(np.linspace(0, 1, self.__num_classes))

        xx = embeddings[:, 0]
        yy = embeddings[:, 1]
        zz = embeddings[:, 2]

        for i in range(self.__num_classes):
            ax.scatter(xx[self.__y_sample==i], yy[self.__y_sample==i], zz[self.__y_sample==i], color=colors[i], label=self.__labels, s=15)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        if self._args.plot_dir:
            plt.savefig(self._args.plot_dir, dpi=600)
        plt.show()

    def run(self):
        args = get_args()
        x_train, y_train = self.__data_loader()

        if args.shuffle:
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

        mask = np.arange(args.num_samples)
        self.__x_sample = x_train[mask].squeeze()
        self.__y_sample = y_train[mask]

        self.__num_classes = len(np.unique(self.__y_sample))
        self.__labels = np.arange(self.__num_classes)
        print 'LABELS NUM: ', len(self.__labels)

        embeddings = None
        if args.compute_embeddings:
            print 'Plotting x_sample shape: {}'.format(self.__x_sample.shape)
            print 'Plotting y_sample shape: {}'.format(self.__y_sample.shape)
            print '-'*70

            x_sample_flat = np.reshape(self.__x_sample, [self.__x_sample.shape[0], -1])

            embeddings = TSNE(n_components=args.num_dimensions, init='pca', verbose=2).fit_transform(x_sample_flat)

        print '-' * 70
        print 'PLOTTING...'

        if args.num_dimensions == 3:
            self._plot3D(embeddings)
        elif args.num_dimensions == 2:
            self._plot2D(embeddings)

        print '-' * 70




if __name__ == '__main__':
    vis = tsne()
    vis.run()
