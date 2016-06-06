from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import nt.testing as tc


class CNNVisu:

    def __init__(self, img_h, img_w,  mod_list):
        self.img_h = img_h
        self.img_w = img_w
        self.mod_list = mod_list
        self.d_max = mod_list[0].in_channels
        for module in self.mod_list:
            getattr(module, "calc_output_size")
            if self.d_max < module.out_channels:
                self.d_max = module.out_channels

    def add_text(self, ax, d, h, w, xcenter, color='blue'):
        if d > 1:
            ax.text(-d/2-3+xcenter, 0, -h/2, "%d" % w, size='medium', color=color)
            ax.text(-d/2-3+xcenter, w/2, 0, "%d" % h, size='medium', color=color)
            ax.text(0+xcenter, -w/2, -h/2-5, "%d" % d, size='medium', color=color)
        else:
            ax.text(xcenter-1, 0, -h/2, "%d" % w, size='small', color=color)
            ax.text(xcenter-1, w/2, 0, "%d" % h, size='small', color=color)

    def conv_area(self, ax, d2, filter_size, xcenter1, xcenter2, layer='ConvLayer'):
        fs_h = filter_size[0]/2
        fs_w = filter_size[1]/2
        z = np.linspace(fs_h, 0, 100)
        x = np.linspace(xcenter1, xcenter2-d2, 100)
        y = np.linspace(fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')
        z = np.linspace(-fs_h, 0, 100)
        x = np.linspace(xcenter1, xcenter2-d2, 100)
        y = np.linspace(-fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')
        z = np.linspace(-fs_h, 0, 100)
        x = np.linspace(xcenter1, xcenter2-d2, 100)
        y = np.linspace(fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')
        z = np.linspace(fs_h, 0, 100)
        x = np.linspace(xcenter1, xcenter2-d2, 100)
        y = np.linspace(-fs_w, 0, 100)
        ax.plot(x, y, z, color="r", linestyle=':')

        z = np.linspace(fs_h, -fs_h, 100)
        x = np.linspace(xcenter1, xcenter1, 100)
        y = np.linspace(fs_w, fs_w, 100)
        ax.plot(x, y, z, color="r")
        z = np.linspace(fs_h, -fs_h, 100)
        x = np.linspace(xcenter1, xcenter1, 100)
        y = np.linspace(-fs_w, -fs_w, 100)
        ax.plot(x, y, z, color="r")
        z = np.linspace(-fs_h, -fs_h, 100)
        x = np.linspace(xcenter1, xcenter1, 100)
        y = np.linspace(-fs_w, fs_w, 100)
        ax.plot(x, y, z, color="r")
        z = np.linspace(fs_h, fs_h, 100)
        x = np.linspace(xcenter1, xcenter1, 100)
        y = np.linspace(-fs_w, fs_w, 100)
        ax.plot(x, y, z, color="r")

        self.add_text(ax, 1, fs_h*2, fs_w*2, xcenter1, color='darkred')

    def layer(self, ax, xcenter, d_max, d2, filter_size, w2, h2):
        d2 = d2/2
        w2 = w2/2
        h2 = h2/2
        depth = [-d2, d2]
        width = [-w2, w2]
        height = [-h2, h2]
        center = [xcenter+2*d_max, 0, 0]
        self.add_text(ax, d2*2, h2*2, w2*2, center[0])
        for s, e in combinations(np.array(list(product(depth, width, height))), 2):
            s = np.array(center)+np.array(s)
            e = np.array(center)+np.array(e)
            # ax.scatter3D(*center, color="r")
            if np.linalg.norm(s-e) == 2*depth[1] or np.linalg.norm(s-e) == 2*width[1] \
                    or np.linalg.norm(s-e) == 2*height[1]:
                # print(zip(s,e))
                ax.plot3D(*zip(s, e), color="k")
        self.conv_area(ax, d2, filter_size, xcenter, center[0])
        d_in = d2*2
        return d_in, center[0]

    def visu(self):
        """
        Example:
        >>> from chainer.links.connection.convolution_2d import CNN2DModule
        >>> from chainer.links.sequential.sequence_cnn import SequenceCNN
        >>> test_list = [
        ...     CNN2DModule((10, 20), 10, 3),
        ...     CNN2DModule(15, 5, 10),
        ...     CNN2DModule((30, 5), 2, 5)
        ... ]
        >>> test_list2 = [
        ...     SequenceCNN(5, 10, 10),
        ...     SequenceCNN(1, 15, 10),
        ...     SequenceCNN(26, 5, 15),
        ...     SequenceCNN(50, 10, 5)
        ... ]
        >>> n = CNNVisu(150, 150, test_list2)
        >>> n.visu()
        >>> m = CNNVisu(60, 60, test_list)
        >>> m.visu()
        """

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("normal")
        # ax.grid(b=False)

        w = self.img_w
        h = self.img_h
        d = self.mod_list[0].in_channels
        depth = [-d/2, d/2]  # x-Axis
        width = [-w/2, w/2]  # y-Axis
        height = [-h/2, h/2]  # z-Axis
        center = [0, 0, 0]
        xcenter = center[0]
        self.add_text(ax, d, h, w, center[0])
        for s, e in combinations(np.array(list(product(depth, width, height))), 2):
            s = np.array(center)+np.array(s)
            e = np.array(center)+np.array(e)
            # ax.scatter3D(*ch_maxenter, color="r")
            if np.linalg.norm(s-e) == 2*depth[1] or np.linalg.norm(s-e) == 2*width[1] \
                    or np.linalg.norm(s-e) == 2*height[1]:
                # print(zip(s,e))
                ax.plot3D(*zip(s, e), color="k")

        for module in self.mod_list:
            h, w = module.calc_output_size(h, w)
            d_out = module.out_channels
            f = module.filter_size
            d, xcenter = self.layer(ax, xcenter, self.d_max, d_out, f, w2=w, h2=h)

        ax.set_xlabel('X axis')
        plt.xticks([])
        plt.axis('off')
        plt.show()
