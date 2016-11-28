from itertools import product, combinations

import matplotlib.pyplot as plt
import numpy as np


class CNNVisu:
    """ creates the cube-visulisation of an CNN
        supports CNN2DModule, SequenceCNN, SequenceRNNCNN,Convolution2D

        Example:
        >>> from chainer.links.connection.convolution_2d import CNN2DModule
        >>> from chainer.links.sequential.sequence_cnn import SequenceCNN, SequenceRNNCNN
        >>> from chainer.links.connection.convolution_2d import Convolution2D
        >>> test_list = [
        ...     CNN2DModule((10, 20), 256, 3),
        ...     CNN2DModule(15, 5, 10),
        ...     CNN2DModule((30, 5), 2, 5)
        ... ]
        >>> test_list2 = [
        ...     SequenceCNN(10, 10, 5),
        ...     SequenceCNN(10, 150, 10),
        ...     SequenceCNN(10, 150, 150),
        ...     SequenceCNN(10, 150, 150, pooling = "max"),
        ...     SequenceCNN(10, 150, 150, pooling = "avg")
        ... ]
        >>> test_list3 = [
        ...     Convolution2D(4,40,7),
        ...     Convolution2D(40, 60, (10,20)),
        ...     Convolution2D(60, 5, 10)
        ...     ]
        >>> test_list4 = [
        ...     SequenceRNNCNN(5, 10, 10),
        ...     SequenceRNNCNN(1, 15, 10),
        ...     SequenceRNNCNN(26, 5, 15)
        ... ]
        >>> n = CNNVisu(150, 150, test_list2)
        >>> n.visu() # with layernames
        >>> n = CNNVisu(150, 150, test_list2)
        >>> n.visu(layer_names=None) # without layernames
    """

    def __init__(self, img_h, img_w, mod_list):
        self.img_h = img_h
        self.img_w = img_w
        self.mod_list = mod_list
        self.conv2d = False
        if mod_list[0].__class__.__name__ == "Convolution2D":
            self.conv2d = True

    def _add_text(self, ax, d, h, w, xcenter, layer_names=None, pooling=None,
                  color='blue'):
        """ Creates textinformations about cubes

        :param ax: matplotlib figure
        :param d: depth of layer
        :param h: height of layer
        :param w: width of layer
        :param xcenter: cube center on x-Axis
        :param layer_names: list with number of current layer[#conv, #max-pooling, #avg-pooling]
        :param pooling: poolinglayer or not
        :param color: text color
        :return: No output
        """
        if d > 1:
            ax.text(-d / 2 + xcenter, 0, -h / 2, "%d" % w, size='medium',
                    color=color, horizontalalignment="center")
            ax.text(-d / 2 + xcenter, w / 2, 0, "%d" % h, size='medium',
                    color=color, horizontalalignment="center")
            ax.text(xcenter, -w / 2, -h / 2 - 5, "%d" % d, size='medium',
                    color=color, horizontalalignment="center")
            if layer_names is not None:
                sgn = layer_names[3] % 2 * 2 - 1
                if pooling is None:
                    if layer_names[3] > 0:
                        ax.text(xcenter, -w / 2, sgn * h / 2 + sgn * 20,
                                "convolution\nlayer %d" % layer_names[0],
                                size='medium', color="green",
                                horizontalalignment="center")
                    else:
                        ax.text(xcenter, -w / 2, -h / 2 - 20, "input\nlayer",
                                size='medium', color="green",
                                horizontalalignment="center")
                elif pooling is "max":
                    ax.text(xcenter, -w / 2, sgn * h / 2 + sgn * 20,
                            "%s-pooling\nlayer %d" % (pooling, layer_names[1]),
                            size='medium', color="green",
                            horizontalalignment="center")
                elif pooling is "avg":
                    ax.text(xcenter, -w / 2, sgn * h / 2 + sgn * 20,
                            "%s-pooling\nlayer %d" % (pooling, layer_names[2]),
                            size='medium', color="green",
                            horizontalalignment="center")
        else:
            ax.text(xcenter, 0, -h / 2, "%d" % w, size='small', color=color,
                    horizontalalignment="center")
            ax.text(xcenter, w / 2, 0, "%d" % h, size='small', color=color,
                    horizontalalignment="center")

    def _conv_area(self, ax, d_out, filter_size, xcenter_in, xcenter_out):
        """ Visualisation of area that gets filtered

        :param ax: matplotlib figure
        :param d_out: depth of output layer
        :param filter_size: filter dimension, can be int or tupel(h,w)
        :param xcenter_in: cube-center of input cube
        :param xcenter_out: cube-center of output cube
        :return: No output
        """
        fs_h = filter_size[0] / 2
        fs_w = filter_size[1] / 2
        z = np.linspace(fs_h, 0, 100)
        x = np.linspace(xcenter_in, xcenter_out - d_out, 100)
        y = np.linspace(fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')
        z = np.linspace(-fs_h, 0, 100)
        x = np.linspace(xcenter_in, xcenter_out - d_out, 100)
        y = np.linspace(-fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')
        z = np.linspace(-fs_h, 0, 100)
        x = np.linspace(xcenter_in, xcenter_out - d_out, 100)
        y = np.linspace(fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')
        z = np.linspace(fs_h, 0, 100)
        x = np.linspace(xcenter_in, xcenter_out - d_out, 100)
        y = np.linspace(-fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')

        z = np.linspace(fs_h, -fs_h, 100)
        x = np.linspace(xcenter_in, xcenter_in, 100)
        y = np.linspace(fs_w, fs_w, 100)
        ax.plot(x, y, z, color="r")
        z = np.linspace(fs_h, -fs_h, 100)
        x = np.linspace(xcenter_in, xcenter_in, 100)
        y = np.linspace(-fs_w, -fs_w, 100)
        ax.plot(x, y, z, color="r")
        z = np.linspace(-fs_h, -fs_h, 100)
        x = np.linspace(xcenter_in, xcenter_in, 100)
        y = np.linspace(-fs_w, fs_w, 100)
        ax.plot(x, y, z, color="r")
        z = np.linspace(fs_h, fs_h, 100)
        x = np.linspace(xcenter_in, xcenter_in, 100)
        y = np.linspace(-fs_w, fs_w, 100)
        ax.plot(x, y, z, color="r")

        self._add_text(ax, 1, fs_h * 2, fs_w * 2, xcenter_in, color='darkred')

    def _layer(self, ax, xcenter, d_min, d_out, filter_size, w_out, h_out,
               layer_names, pooling=None):
        """ Generates cubevisualisation of new layer.

        :param ax: matplotlib figure
        :param xcenter: center of cube on x-Axis
        :param d_min: min distance between two cubes
        :param d_out: depth of generating layer
        :param filter_size: filter dimension, can be int or tupel(h,w)
        :param w_out: width of generating  layer
        :param h_out: height of generating layer
        :param layer_names: list with number of current layer[#conv, #max-pooling, #avg-pooling]
        :param pooling: poolinglayer or not
        :return: depth of generated cube, center of generated cube
        """
        d_out = d_out / 2
        w_out = w_out / 2
        h_out = h_out / 2
        depth = [-d_out, d_out]
        width = [-w_out, w_out]
        height = [-h_out, h_out]
        center = [xcenter + d_min / 2 + 50, 0, 0]
        self._add_text(ax, d_out * 2, h_out * 2, w_out * 2, center[0],
                       layer_names, pooling=pooling)
        for s, e in combinations(np.array(list(product(depth, width, height))),
                                 2):
            s = np.array(center) + np.array(s)
            e = np.array(center) + np.array(e)
            if np.linalg.norm(s - e) == 2 * depth[1] or np.linalg.norm(
                            s - e) == 2 * width[1] \
                    or np.linalg.norm(s - e) == 2 * height[1]:
                ax.plot3D(*zip(s, e), color="k", alpha=0.2)
        self._conv_area(ax, d_out, filter_size, xcenter, center[0])
        d_in = d_out * 2
        return d_in, center[0]

    def visu(self, layer_names=[0, 0, 0, 0]):
        """

        :param layer_names: (#conv, #max-pool, #avg-pool, #layer) or None if layernames not needed
        :return: No Output
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("normal")

        w = self.img_w
        h = self.img_h
        if self.conv2d is False:
            d = self.mod_list[0].in_channels
        else:
            d = self.mod_list[0].W.data.shape[1]
        depth = [-d / 2, d / 2]  # x-Axis
        width = [-w / 2, w / 2]  # y-Axis
        height = [-h / 2, h / 2]  # z-Axis
        center = [0, 0, 0]
        xcenter = center[0]  # center of cube on x-Axis
        # create first layer/ input img:
        self._add_text(ax, d, h, w, center[0], layer_names)
        for s, e in combinations(np.array(list(product(depth, width, height))),
                                 2):
            s = np.array(center) + np.array(s)
            e = np.array(center) + np.array(e)
            if np.linalg.norm(s - e) == 2 * depth[1] or np.linalg.norm(
                            s - e) == 2 * width[1] \
                    or np.linalg.norm(s - e) == 2 * height[1]:
                ax.plot3D(*zip(s, e), color="k", alpha=0.2)

        # create following layers/ convolution- and poolinglayers:
        for module in self.mod_list:
            if self.conv2d is False:
                d_out = module.out_channels
                d_in = module.in_channels
                filter_size = module.filter_size
                pooling = module.pooling
            else:
                d_out = module.W.data.shape[0]
                d_in = module.W.data.shape[1]
                filter_size = (module.W.data.shape[2], module.W.data.shape[3])
                pooling = None
            h, w = module._calc_cnn_output_size(h, w)
            if layer_names is not None:
                if pooling is None:
                    layer_names[0] = layer_names[0] + 1
                    layer_names[3] = layer_names[3] + 1
                elif pooling is "max":
                    layer_names[1] = layer_names[1] + 1
                    layer_names[3] = layer_names[3] + 1
                else:
                    layer_names[2] = layer_names[2] + 1
                    layer_names[3] = layer_names[3] + 1
            d, xcenter = self._layer(ax, xcenter, d_in + d_out, d_out,
                                     filter_size,
                                     w_out=w, h_out=h, layer_names=layer_names,
                                     pooling=pooling)

        ax.set_xlabel('X axis')
        plt.xticks([])
        plt.axis('off')
        plt.show()
