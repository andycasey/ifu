import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

class DataCube(object):

    def __init__(self, path):

        self.path = os.path.realpath(os.path.abspath(path))
        self.image = fits.open(path)


    @property
    def dispersion(self):

        hdu, axis = (0, 3)
        header = self.image[hdu].header
        pixel = np.arange(self.image[hdu].data.shape[::-1][axis - 1])

        return header["CRVAL3"] + header["CDELT3"] * (pixel - header["CRPIX3"])


    @property
    def flux(self):
        return self.image[0].data


    def polar_coordinates(self, x_centroid=None, y_centroid=None):
        """
        Return polar coordinates for every pixel.
        """
        hdu = 0

        xs, ys = self.image[hdu].data.shape[1:]
        x_centroid, y_centroid = (x_centroid or xs/2., y_centroid or ys/2.)

        xp, yp = (np.arange(xs), np.arange(ys))
        x, y = np.meshgrid(xp - x_centroid, yp - y_centroid)

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) * 180.0/np.pi

        return (r, theta)



    def inspect(self):

        data = self.image[0].data

        # Create interactive figure.
        fig = plt.figure(figsize=(12.5, 3))

        gs = GridSpec(1, 2, width_ratios=(4, 1))

        ax_spectrum = fig.add_subplot(gs[0])
        ax_cube = fig.add_subplot(gs[1])

        ax_cube.imshow(np.nansum(data, axis=0).T, cmap="plasma", norm=LogNorm(),
                       interpolation="none", aspect=1.0,
                       extent=(0, data.shape[1], 0, data.shape[1]))
        ax_cube.grid(lw=0.25, c="#FFFFFF", linestyle="-", alpha=0.5)
        ax_cube.set_xticks(np.arange(data.shape[1]))
        ax_cube.set_yticks(np.arange(data.shape[2]))



        ax_cube.set_xticklabels([])
        ax_cube.set_yticklabels([])
        ax_cube.xaxis.set_tick_params(width=0)
        ax_cube.yaxis.set_tick_params(width=0)

        spectrum_plot, = ax_spectrum.plot([], [], c="k")

        square = lambda i, j, left=0, right=1: np.atleast_2d([
            [i - left, j - left],
            [i + right, j - left],
            [i + right, j + right],
            [i - left, j + right],
            [i - left, j - left]
        ]).T

        selected_spaxel, = ax_cube.plot([], [], c="k")

        def update_spectrum_plot(event):
            if event.xdata is None or event.ydata is None or event.inaxes != ax_cube:
                return None

            i, j = int(event.xdata), int(event.ydata)

            flux = data[:, i, j]

            spectrum_plot.set_data(np.vstack([self.dispersion, flux]))
            selected_spaxel.set_data(square(i, j))

            if np.any(np.isfinite(flux)):
                ax_spectrum.set_xlim(self.dispersion[0], self.dispersion[-1])
                ax_spectrum.set_ylim(0, 1.05 * np.nanmax(flux))

            fig.canvas.draw()


        cid = fig.canvas.mpl_connect('button_release_event', update_spectrum_plot)

        fig.tight_layout()

        return fig


if __name__ == "__main__":


    DATA_PATH = "../sami/"
    path = "227607/227607_spectrum_re_red.fits"
    path = "91926_cube_red.fits"

    cube = DataCube(os.path.join(DATA_PATH, path))
    fig = cube.inspect()

    r, theta = cube.polar_coordinates()

    fig, ax = plt.subplots()
    im = ax.imshow(r)
    plt.colorbar(im)


    fig, ax = plt.subplots()
    im = ax.imshow(theta)
    plt.colorbar(im)

