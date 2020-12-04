"""
Various tools to help in simulation methods.

Code for the article "Monte Carlo Methods for the Neutron Transport Equation.

By A. Cox, A. Kyprianou, S. Harris, M. Wang.

Thi sfile contains the code to produce the plots in the case of the 2D version
of the NTE.

MIT License

Copyright (c) Alexander Cox, 2020.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SimulateFunction:
    """Simulate a (possibly array valued) function, rep times.

    Note, expects fun to return a np array with one dimension.
    """

    def __init__(self, fun, rep, ProgBar=False):
        x = fun()
        self.z = np.zeros((rep, len(x)))
        self.z[0, :] = x

        if ProgBar:
            for i in tqdm(range(rep-1)):
                self.z[i+1, :] = fun()
        else:
            for i in range(rep-1):
                self.z[i+1, :] = fun()

    def mean(self):
        """Return Mean."""
        return np.mean(self.z, 0)

    def var(self):
        """Return Variance."""
        return np.var(self.z, 0)

    def sensitivity_to_outliers(self, fun2=np.mean):
        """Return a measure of how sensitive the mean/fun2 is to outliers."""
        # NB assumes fun2 will act on matrix with rows corresponding to
        #   repetitions, columns corresponding to different data points.
        #   fun2 should not worry about order of repetitions.
        (n, m) = np.shape(self.z)
        h = np.zeros((n, m))
        bar = self.mean()
        comp = fun2(self.z)
        for i in range(n):
            x = np.vstack((self.z[0:i, :], self.z[(i+1):n, :], bar))
            h[i, :] = np.abs(fun2(x)-comp)
        return np.max(h, 0)

    def plot_mean(self, vals=None):
        """Plot the mean values."""
        if vals is None:
            vals = range(self.z.shape[1])

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        axes.plot(vals, self.mean())

        plt.draw()

    def plot_var(self, vals=None):
        """Plot the variance."""
        if vals is None:
            vals = range(self.z.shape[1])

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        axes.plot(vals, self.var())

        plt.draw()

    def plot_violin(self, vals=None, num_vio=5, log=False, title=""):
        """Produce a violin plot of the data."""
        n = self.z.shape[1]
        eps = 1e-8

        if vals is None:
            vals = range(n)

        k = int(n/num_vio)

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        x = [j for j in range(0, n, k)]
        if log:
            y = [np.log(np.maximum(self.z[:, j], eps)) for j in range(0, n, k)]
        else:
            y = [self.z[:, j] for j in range(0, n, k)]

        axes.violinplot(y, x,
                        points=60, widths=2.7, showmeans=True,
                        showextrema=True, showmedians=True, bw_method=0.5)
        axes.set_title(title)

        plt.draw()

    def plot_box(self, vals=None, num_vio=5, log=False, title=""):
        """Produce a box plot of the data."""
        n = self.z.shape[1]
        eps = 1e-8

        if vals is None:
            vals = range(n)

        k = int(n/num_vio)

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        # x = [j for j in range(0,n,k)]

        # y = [self.z[:,j] for j in range(0,n,k)]

        if log:
            y = [np.log(np.maximum(self.z[:, j], eps)) for j in range(0, n, k)]
        else:
            y = [self.z[:, j] for j in range(0, n, k)]

        axes.boxplot(y)
        axes.set_title(title)

        plt.draw()
