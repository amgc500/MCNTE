"""
Computes the Fixed Point of the Eigenvalue Equation.

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


class Eigenvalue:
    """
    Compute the eigenvalue for the 1D NTE by solving a fixed point equation.

    Numerically find solution to x/sin(x)=c for c>1 and x/sinh(x)=c for c<1,
    then use this solution to compute the leading eigenvalue.
    """

    def __init__(self, a, b, alpha, v, beta, K):
        c = (b-a)*alpha/v
        self.theta = 1/c
        self.L = (b-a)/2
        self.cent = (a+b)/2
        if c == 1:
            self.value = beta - 2 * alpha
        if c > 1:
            x0 = 0
            x1 = np.pi
            for k in range(K):
                mid = (x0 + x1)/2
                if mid/np.sin(mid)-c >= 0:
                    x1 = mid
                else:
                    x0 = mid
            self.sol = (x0+x1)/2
            self.res = self.sol/np.sin(self.sol)-c
            if self.sol > np.pi/2:
                sgn = -1
            else:
                sgn = 1
            self.value = (beta - alpha - sgn *
                          np.sqrt(alpha * alpha -
                                  np.square((v * self.sol)/(b - a))))
        elif c > 0 and c < 1:
            x = 1
            for k in range(K):
                f = x/np.sinh(x) - c
                df = (np.sinh(x) - x * np.cosh(x))/(np.sinh(x) * np.sinh(x))
                x -= f/df
            self.sol = x
            self.res = x/np.sinh(x) - c
            self.value = (beta - alpha
                          - np.sqrt(alpha * alpha +
                                    np.square((v * self.sol)/(b - a))))
        else:
            print("\n non valid input")

        # Normalise varphi:
        N = 100
        x = np.linspace(a, b, N)
        self.norm_const = np.mean(self.phi(x)*self.phi(np.flip(x)))*4*self.L

    def phi(self, r):
        """Return the function phi."""
        r_cent = r - self.cent
        if self.theta == 1:
            return 1-r_cent/self.L
        elif self.theta > 1:
            return np.sinh(self.sol*(self.L-r_cent) /
                           (2*self.L))/np.sinh(self.sol/2)
        elif self.theta < 1:
            return np.sin(self.sol*(self.L-r_cent) /
                          (2*self.L))/np.sin(self.sol/2)

    def varphi(self, r, v):
        """Return the function varphi."""
        return self.phi(r * np.sign(v))

    def varphi_tilde(self, r, v):
        """Return the function varphi_tilde."""
        return self.phi(-r * np.sign(v))/self.norm_const


class Eigenfunction:
    """
    Define several classes of Eigenfunction.

    Class defines linear, minimum and product versions which approximate the
        true eigenfunction, varphi.

    Requires input parameters:
        a and b, representing the left and right endpoints of the domain
        alpha, representing the scatter rate in the domain,
        V_0, the velocity
        beta, the branching rate in the domain. (Not currently used!)
        y, the corresponding true eigenvalue.
    """

    def __init__(self, a, b, alpha, V_0, beta, y):
        c = (b - a) * alpha / np.abs(V_0)
        if c == 1 and V_0 > 0:
            self.varphi = lambda arg: (b - arg) / b
            self.linear = self.varphi
            self.mini = self.linear
            self.product = lambda arg: ((b - arg) * (arg - a + V_0 / alpha) /
                                        (b * (-a + V_0 / alpha)))
        elif c == 1 and V_0 < 0:
            self.varphi = lambda arg: (arg - a) / (-a)
            self.linear = self.varphi
            self.mini = self.linear
            self.product = lambda arg: ((arg - a) * (b - arg - V_0 / alpha)
                                        / (-a * (b - V_0 / alpha)))
        elif c > 1 and V_0 > 0:
            u = np.sin(y.sol/2)
            self.varphi = lambda arg: np.sin(y.sol*(b-arg)/(b-a))/u
            self.linear = lambda arg: (b-arg)/b
            self.mini = lambda arg: np.amin([b-arg, arg-a+V_0/alpha])/b
            self.product = lambda arg: ((b-arg)*(arg-a+V_0/alpha)
                                        / (b*(-a+V_0/alpha)))
        elif c > 1 and V_0 < 0:
            u = np.sin(y.sol/2)
            self.varphi = lambda arg: np.sin(y.sol*(arg-a)/(b-a))/u
            self.linear = lambda arg: (arg-a)/(-a)
            self.mini = lambda arg: np.amin([arg-a, b-arg-V_0/alpha])/(-a)
            self.product = lambda arg: ((arg-a)*(b-arg-V_0/alpha)
                                        / (-a*(b-V_0/alpha)))
        elif c < 1 and V_0 > 0:
            u = np.sinh(y.sol/2)
            self.varphi = lambda arg: np.sinh(y.sol*(b-arg)/(b-a))/u
            self.linear = lambda arg: (b-arg)/b
            self.mini = self.linear
            self.product = lambda arg: ((b-arg)*(arg-a+V_0/alpha)
                                        / (b*(-a+V_0/alpha)))
        elif c < 1 and V_0 < 0:
            u = np.sinh(y.sol/2)
            self.varphi = lambda arg: np.sinh(y.sol*(arg-a)/(b-a))/u
            self.linear = lambda arg: (arg-a)/(-a)
            self.mini = self.linear
            self.product = lambda arg: ((arg-a)*(b-arg-V_0/alpha)
                                        / (-a*(b-V_0/alpha)))
        else:
            print("\n No valid input")
