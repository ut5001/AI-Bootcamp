import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mu = numpy.matrix([[175], [80]])
# Sigma = numpy.matrix([[100, 0],[0, 25]])
Sigma = numpy.matrix([[100, 25],[25, 25]])


def plot_gauss2d(mu, Sigma):
    X = numpy.arange(145, 205.2, 0.2)
    Y = numpy.arange(65, 95.1, 0.1)
    X, Y = numpy.meshgrid(X, Y)
    siginv = numpy.linalg.inv(Sigma)
    xmmu = X - mu[0, 0]
    ymmu = Y - mu[1, 0]
    xts1 = xmmu * siginv[0, 0] + ymmu * siginv[1, 0]
    xts2 = xmmu * siginv[1, 0] + ymmu * siginv[1, 1]
    d = -1 / 2 * (xts1 * xmmu + xts2 * ymmu)
    Z = 1 / (2 * numpy.pi) / numpy.sqrt(numpy.linalg.det(Sigma)) * numpy.exp(d)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(60, 280)  # Elevation, Azimuth
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


plot_gauss2d(mu, Sigma)