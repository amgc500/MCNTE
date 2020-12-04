"""Code for the 1D Case of the NTE.

Classes SubPathCons and PathsCons deal with the case of constant scattering.

Classes SubPathLinear and PathsLinear deal with the case of a linear
scattering function.

Classes SubPathSpine and PathsSpine deal with the case where the scattering
function is given by the leading eigenfunction.

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
from scipy.integrate import quad


class SubPathCons:
    """Define a subpath of a neutron."""

    def __init__(self, tstart, xstart, vstart, tfin, srate, a, b):
        self.tstart = tstart

        self.xstart = xstart
        self.vstart = vstart
        self.tfin = tfin
        # tfin is lifetime of the particle. Initially ends at final time.

        self.turn = [[self.tstart, self.xstart, self.vstart]]
        # turn is a list of (time, position, direction) where the particle
        # changes directions, including the starting and ending points.
        self.scatter(self.tfin, srate, a, b)

    def scatter(self, tfin, srate, a, b):
        """Generate the list turn."""
        temp = self.tstart
        position = self.xstart
        speed = self.vstart
        if speed > 0:
            zeta = (b-position)/speed
        else:
            zeta = (a-position)/speed
        alive = 1
        while temp < tfin and alive == 1:
            Exp = -np.log(np.random.uniform())/srate
            # Constant scattering
            temp += Exp
            if np.amin([temp, temp-Exp+zeta]) < tfin:
                if Exp >= zeta:
                    alive = 0
                    self.tfin = temp-Exp+zeta
                    self.turn.append([self.tfin, position + speed*zeta, speed])

                else:
                    position += speed * Exp
                    speed = -speed
                    if speed > 0:
                        zeta = (b-position)/speed
                    else:
                        zeta = (a-position)/speed
                    self.turn.append([temp, position, speed])

        if alive == 1:
            self.turn.append([tfin, self.turn[-1][1] +
                              (tfin - self.turn[-1][0]) * self.turn[-1][2],
                              self.turn[-1][2]])

        # Add the final point
        return self.turn

    def where(self, temps):
        """Return the position and direction of particle at a given time."""
        if temps < self.tstart or temps > self.tfin:
            print(self.tstart, self.tfin, temps)
            # Error: Some tfin < tstart, find out why!!
            return [-100, -100]
        if len(self.turn) <= 1:
            return [self.xstart+(temps-self.tstart)*self.vstart, self.vstart]
        i = 0
        while i < len(self.turn)-1:
            if temps >= self.turn[i][0] and temps <= self.turn[i+1][0]:
                return [self.turn[i][1] + (temps - self.turn[i][0]) *
                        self.turn[i][2], self.turn[i][2]]
                break
            else:
                i += 1
        print("\n Oops, something wrong...")
        return [-200, -200]

    def dress(self, rate, branch_times):
        """Return positions and times for branching, Add to branch_times."""
        tt = self.tstart
        while tt < self.tfin:
            Exp = np.random.exponential(1/rate)
            tt += Exp
            if tt < self.tfin:
                branch_times.append([tt, self.where(tt)[0], self.where(tt)[1]])

    def is_alive(self, time_val):
        """Check if the particle is alive at a given time."""
        return (self.tstart <= time_val < self.tfin)

    def was_alive(self, time_val):
        """Check if the particle was alive before a given time."""
        return (self.tstart <= time_val)

    def count_scatters(self, time_val):
        """Count number of scatter events up to given time."""
        if self.was_alive(time_val):
            i = 0
            while (i < len(self.turn) and self.turn[i][0] <= time_val):
                i += 1
            return i
        else:
            return 0


class PathsCons:
    """Define a collection of neutron trajectories."""

    def __init__(self, tstart, tfin, xstart, vstart, beta, srate, a, b,
                 max_particles=50000):
        self.branch_times = []
        self.trajectories = []
        self.tstart = tstart
        self.xstart = xstart
        self.vstart = vstart

        # ttstart = time()
        self.trajectories.append(SubPathCons(tstart, xstart, vstart, tfin,
                                             srate, a, b))
        if beta > 0:
            self.trajectories[0].dress(beta, self.branch_times)

            while (len(self.trajectories) < max_particles and
                   len(self.branch_times) > 0):
                temps, start, veloc = self.branch_times.pop()
                # list.pop removes and returns the last element
                self.trajectories.append(SubPathCons(temps, start, veloc,
                                                     tfin, srate, a, b))
                self.trajectories[-1].dress(beta, self.branch_times)

        if len(self.trajectories) >= max_particles:
            print("\nWARNING: Maximum number of particles exceeded.\n")

        # ttend = time()
        # print("time consumed for simulate the whole system", ttend - ttstart)

    def plot(self):
        """Plot the trajectory of a particle."""
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories))))

        for x in self.trajectories:
            c = next(color)
            t_local = [row[0] for row in x.turn]
            traj = []
            for ts in t_local:
                traj.append([x.where(ts)[0]])
            axes.plot(t_local, traj, c=c)

        plt.draw()

    def count(self, time_val):
        """Return number of particles alive at given time."""
        return sum(1 for x in self.trajectories if x.is_alive(time_val))

    def count_scatters(self, time_val):
        """Return number of scatters up to given time."""
        return sum(x.count_scatters(time_val) for x in self.trajectories)

    def count_was_alive(self, time_val):
        """Return number of particles alive at given time."""
        return sum(1 for x in self.trajectories if x.was_alive(time_val))


class DistF:
    """Compute the dist. fn. of a linear rate function alpha(b-x)/(x-a)."""

    def __init__(self, v, x, alpha, a, b):
        # v is the current speed of the particle and x its position.
        if v > 0:
            zeta = (b-x)/v
        else:
            zeta = (a-x)/v
        t = np.linspace(0, 0.99*zeta, 100)
        self.grid = t
        self.f = np.zeros(len(self.grid))
        if v > 0:
            for i in range(len(self.grid)):
                self.f[i] = 1 - np.exp(alpha * t[i] + (b - a) * alpha / v
                                       * np.log((b - x - v * t[i]) / (b - x)))
        else:
            for i in range(len(self.grid)):
                self.f[i] = 1 - np.exp(alpha * t[i] - (b - a) * alpha / v
                                       * np.log((x - a + v * t[i]) / (x - a)))

    def plot(self):
        """Plot the Dist. Fn."""
        fig = plt.figure()
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(self.grid, self.f)
        plt.draw()


class SubPathLinear:
    """Define subpath of a neutron."""

    def __init__(self, tstart, xstart, vstart,
                 tfin, alpha, a, b):
        # The scattering function is of the form:
        #   (alpha * (b-x)/x-a, alpha*(x-a)/(b-x))

        self.tstart = tstart

        self.xstart = xstart
        self.vstart = vstart
        self.tfin = tfin  # tfin is lifetime of the particle.

        self.turn = [[self.tstart, self.xstart, self.vstart]]
        # turn is a list of (time, position, direction) where the particle
        # changes directions, including the starting and ending points.

        self.scatter(self.tfin, alpha, a, b)

    def scatter(self, tfin, alpha, a, b):
        """Generate the list turn."""
        temp = self.tstart
        position = self.xstart
        speed = self.vstart
        if speed > 0:
            zeta = (b-position)/speed
        else:
            zeta = (a-position)/speed
        alive = 1
        while temp < tfin and alive == 1:
            U = np.random.uniform()
            Dist = DistF(speed, position, alpha, a, b)
            index = np.searchsorted(Dist.f, U)
            Exp = Dist.grid[np.amin([index, len(Dist.f) - 1])]

            # Linear scattering function
            temp += Exp
            if np.amin([temp, temp-Exp+zeta]) < tfin:
                if Exp >= zeta:
                    alive = 0
                    self.tfin = temp-Exp+zeta
                    self.turn.append([self.tfin, position + speed * zeta,
                                      speed])
                else:
                    position += speed * Exp
                    speed = -speed
                    if speed > 0:
                        zeta = (b-position)/speed
                    else:
                        zeta = (a-position)/speed
                    self.turn.append([temp, position, speed])
        if alive == 1:
            self.turn.append([tfin,
                              self.turn[-1][1] + (tfin - self.turn[-1][0])
                              * self.turn[-1][2], self.turn[-1][2]])
        # Add the final point
        return self.turn

    def where(self, temps):
        """Return the position and direction of particle at a given time."""
        if temps < self.tstart or temps > self.tfin:
            print(self.tstart, self.tfin, temps)
            return [-100, -100]
        if len(self.turn) <= 1:
            return [self.xstart + (temps - self.tstart) * self.vstart,
                    self.vstart]
        i = 0
        while i < len(self.turn)-1:
            if temps >= self.turn[i][0] and temps <= self.turn[i+1][0]:
                return [self.turn[i][1]+(temps - self.turn[i][0])
                        * self.turn[i][2], self.turn[i][2]]
                break
            else:
                i += 1
        print("\n Oops, something wrong...")
        return [-200, -200]

    def dress(self, rate, branch_times):
        """Return positions and times for branching, add to branch_times."""
        tt = self.tstart
        while tt < self.tfin and rate > 0:
            Exp = np.random.exponential(1/rate)
            tt += Exp
            if tt < self.tfin:
                branch_times.append([tt, self.where(tt)[0], self.where(tt)[1]])

    def is_alive(self, time_val):
        """Check if the particle is alive at a given time."""
        return (self.tstart <= time_val < self.tfin)


class PathsLinear:
    """Define a collection of neutron trajectories."""

    def __init__(self, tstart, tfin, xstart, vstart, beta, srate,
                 a, b, max_particles=5000):
        self.branch_times = []
        self.trajectories = []
        self.tstart = tstart
        self.xstart = xstart
        self.vstart = vstart
        self.alpha = srate
        self.beta = beta
        self.a = a
        self.b = b

        self.trajectories.append(SubPathLinear(tstart, xstart, vstart, tfin,
                                               srate, a, b))
        self.trajectories[0].dress(beta, self.branch_times)

        while (len(self.trajectories) < max_particles and
               len(self.branch_times) > 0):
            temps, start, veloc = self.branch_times.pop()
            # list.pop removes and returns the last element
            self.trajectories.append(SubPathLinear(temps, start, veloc, tfin,
                                                   srate, a, b))
            self.trajectories[-1].dress(beta, self.branch_times)

        if len(self.trajectories) >= max_particles:
            print("\nWARNING: Maximum number of particles exceeded.\n")

    def plot(self):
        """Plot the particle trajectory."""
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories))))

        for x in self.trajectories:
            c = next(color)
            t_local = [row[0] for row in x.turn]
            traj = []
            for ts in t_local:
                traj.append([x.where(ts)[0]])
            axes.plot(t_local, traj, c=c)

        plt.draw()

    def count(self, time_val):
        """Return number of particles alive."""
        return sum(1 for x in self.trajectories if x.is_alive(time_val))

    def integral(self, time_start, times):
        """Compute the weight integral. Only works for h_+=b-x and h_-=x-a."""
        Integ = -2 * self.alpha * (times - time_start)
        v = np.absolute(self.vstart)
        for x in self.trajectories:
            i = 0
            while i < len(x.turn)-1 and x.turn[i+1][0] < times:
                s = x.turn[i]
                ss = x.turn[i+1]
                if s[0] > time_start:
                    if s[2] > 0:
                        Integ += ((self.alpha*(self.b - self.a)/v - 1) *
                                  np.log(np.amin([10000, (self.b - s[1]) /
                                                  (self.b - ss[1])])))
                    else:
                        Integ += ((self.alpha*(self.b - self.a)/v - 1) *
                                  np.log(np.amin([10000, (s[1]-self.a) /
                                                  (ss[1] - self.a)])))

                elif s[0] < time_start and ss[0] > time_start:
                    if s[2] > 0:
                        Integ += ((self.alpha * (self.b - self.a)/v - 1) *
                                  np.log(np.amin(
                                      [10000, (self.b -
                                               x.where(time_start)[0]) /
                                       (self.b - ss[1])])))
                    else:
                        Integ += ((self.alpha*(self.b - self.a)/v - 1) *
                                  np.log(np.amin(
                                      [10000, (x.where(time_start)[0] -
                                               self.a)/(ss[1] - self.a)])))

                i += 1
            if x.turn[i][0] < times:
                if x.turn[i][2] > 0:
                    Integ += ((self.alpha*(self.b - self.a)/v - 1) *
                              np.log(np.amin([10000,
                                              (self.b - x.turn[i][1]) /
                                              (self.b - x.where(times)[0])])))
                else:
                    Integ += ((self.alpha*(self.b - self.a)/v - 1) *
                              np.log(np.amin([10000,
                                              (x.turn[i][1] - self.a) /
                                              (x.where(times)[0] - self.a)])))

        return np.exp(Integ)


class IntPhi:
    """Numerically compute the integral for the simulation of scattering times.

    Integral is: int_{-L}^x alpha*phi_-(u)/(v*phi_+(u))du,
    which is then used to simulate scattering times.
    """

    def __init__(self, alpha, v, L, y, N=500):

        c = 2 * L * alpha / v
        if c == 1:
            Const = c
            sfun1 = lambda t: (t + L) / (L - t)

        elif c < 1:
            Const = c * np.sinh(y) / y
            sfun1 = lambda t: ((np.exp(t*y/(2*L))
                               - np.exp(-t*y/(2*L)-y)) /
                               (np.exp(-t*y/(2*L)) - np.exp(t*y/(2*L)-y)))
        else:
            Const = c * np.sin(y) / y
            sfun1 = lambda t: ((np.sin(t*y/(2*L)) + np.tan(y/2) *
                                np.cos(t*y/(2*L))) / (np.tan(y/2) *
                                                      np.cos(t*y/(2*L))
                                                      - np.sin(t*y/(2*L))))

        x = np.linspace(-L, L, N)
        x = x[1:-1]
        self.grid = x
        I1 = np.zeros(len(self.grid))
        I1[0] = Const*np.log((2*L)/(L-x[0]))
        I2 = np.zeros(len(self.grid))
        integrand = lambda t: sfun1(t) - v * Const / (alpha * (L - t))
        I2[0] = quad(integrand, -L, x[0])[0]

        for i in range(len(self.grid)-1):
            I1[i+1] = Const*np.log((2*L)/(L-x[i+1]))
            I2[i+1] = I2[i]+alpha/v*quad(integrand, x[i], x[i+1])[0]
        self.int = I1+I2

    def plot(self):
        """Plot the integral function."""
        fig = plt.figure()
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(self.grid, self.int)
        plt.draw()


class DistG:
    """Simulate the next scattering time."""

    def __init__(self, alpha, v, L, y, x, V, I):
        # x is the position of the particle, V its velocity

        if V > 0:
            index = np.amin([len(I.grid)-1, np.searchsorted(I.grid, x)])
            x_0 = I.grid[index]
            tt = x_0*np.ones(len(I.grid)-index)
            self.timegrid = (I.grid[index:]-tt)/V
            self.G = np.zeros(len(I.grid)-index)
            for i in range(index, len(I.grid)):
                self.G[i-index] = 1 - np.exp(-I.int[i]+I.int[index])
            U = np.random.uniform()
            tIndex = np.amin([len(self.timegrid)-1,
                              np.searchsorted(self.G, U)])
            self.exp = self.timegrid[tIndex]
        else:
            # Using the symmetry of the system, replace x by -x
            index = np.amin([len(I.grid)-1, np.searchsorted(I.grid, -x)])
            x_0 = I.grid[index]
            tt = x_0*np.ones(len(I.grid)-index)
            self.timegrid = (I.grid[index:]-tt)/(-V)
            self.G = np.zeros(len(I.grid)-index)
            for i in range(index, len(I.grid)):
                self.G[i-index] = 1 - np.exp(-I.int[i]+I.int[index])
            U = np.random.uniform()
            tIndex = np.amin([len(self.timegrid)-1,
                              np.searchsorted(self.G, U)])
            self.exp = self.timegrid[tIndex]

    def plot(self):
        """Plot the Distribution Function."""
        fig = plt.figure()
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(self.timegrid, self.G)
        plt.draw()


class SubPathSpine:
    """Define a subpath of a neutron."""

    def __init__(self, tstart, xstart, vstart, tfin, alpha, a, b, y, dist):
        # IF IsLinear is TRUE, the scattering function is of the form
        #   (alpha * (b-x)/x-a, alpha*(x-a)/(b-x))
        # Otherwise, it is given by (sfun1, sfun2)

        self.tstart = tstart

        self.xstart = xstart
        self.vstart = vstart
        self.tfin = tfin  # tfin is lifetime of the particle.

        self.turn = [[self.tstart, self.xstart, self.vstart]]
        # turn is a list of (time, position, direction)
        #  where the particle changes directions,
        #  including the starting and ending points.
        self.scatter(self.tfin, alpha, a, b, y, dist)

    def scatter(self, tfin, alpha, a, b, y, dist):
        """Generate the list turn."""
        temp = self.tstart
        position = self.xstart
        speed = self.vstart
        if speed > 0:
            zeta = (b-position)/speed
        else:
            zeta = (a-position)/speed
        alive = 1
        while temp < tfin and alive == 1:
            Exp = DistG(alpha, self.vstart, b, y, position, speed, dist).exp
            temp += Exp
            if np.amin([temp, temp - Exp + zeta]) < tfin:
                if Exp >= zeta:
                    alive = 0
                    self.tfin = temp-Exp+zeta
                    self.turn.append([self.tfin, position + speed * zeta,
                                      speed])
                else:
                    position += speed * Exp
                    speed = -speed
                    if speed > 0:
                        zeta = (b-position)/speed
                    else:
                        zeta = (a-position)/speed
                    self.turn.append([temp, position, speed])
        if alive == 1:
            self.turn.append([tfin, (self.turn[-1][1] +
                                     (tfin - self.turn[-1][0]) *
                                     self.turn[-1][2], self.turn[-1][2])])
        # Add the final point
        return self.turn

    def where(self, temps):
        """Return the position and direction of particle at a given time."""
        if temps < self.tstart or temps > self.tfin:
            print(self.tstart, self.tfin, temps)
            # Some tfin < tstart, return a message in this case!!
            return [-100, -100]
        if len(self.turn) <= 1:
            return [self.xstart+(temps-self.tstart)*self.vstart, self.vstart]
        i = 0
        while i < len(self.turn)-1:
            if temps >= self.turn[i][0] and temps <= self.turn[i+1][0]:
                return [self.turn[i][1] + (temps-self.turn[i][0])
                        * self.turn[i][2], self.turn[i][2]]
                break
            else:
                i += 1
        print("\n Oops, something wrong...")
        return [-200, -200]

    def dress(self, rate, branch_times):
        """Return positions and times for branching, add to branch_times."""
        tt = self.tstart
        while tt < self.tfin and rate > 0:
            Exp = np.random.exponential(1/rate)
            tt += Exp
            if tt < self.tfin:
                branch_times.append([tt, self.where(tt)[0], self.where(tt)[1]])

    def is_alive(self, time_val):
        """Check if the particle is alive at a given time."""
        return (self.tstart <= time_val < self.tfin)


class PathSpine:
    """Define a collection of neutron trajectories."""

    def __init__(self, tstart, tfin, xstart, vstart, beta, srate, a, b, y,
                 max_particles=5000):
        self.branch_times = []
        self.trajectories = []
        self.tstart = tstart
        self.xstart = xstart
        self.vstart = vstart
        self.alpha = srate
        self.beta = beta
        self.a = a
        self.b = b

        Integ = IntPhi(srate, vstart, b, y)
        self.trajectories.append(SubPathSpine(tstart, xstart, vstart, tfin,
                                              srate, a, b, y, Integ))
        self.trajectories[0].dress(beta, self.branch_times)

        while (len(self.trajectories) < max_particles
               and len(self.branch_times) > 0):
            temps, start, veloc = self.branch_times.pop()
            # list.pop removes and returns the last element
            self.trajectories.append(SubPathSpine(temps, start, veloc, tfin,
                                                  srate, a, b, y, Integ))
            self.trajectories[-1].dress(beta, self.branch_times)

        if len(self.trajectories) >= max_particles:
            print("\nWARNING: Maximum number of particles exceeded.\n")

    def plot(self):
        """Plot the trajectory of the particle."""
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories))))

        for x in self.trajectories:
            c = next(color)
            t_local = [row[0] for row in x.turn]
            traj = []
            for ts in t_local:
                traj.append([x.where(ts)[0]])
            axes.plot(t_local, traj, c=c)

        plt.draw()


class DistG:
    """Compute the distribution function of a Minimum-tyoe rate fucntion """
    def __init__(self, v, x, alpha, L, N=500):
        # v is the current speed of the particle and x its position. 
        if v < 0:
            print("\n No valid input")
        else:
            zeta = (L-x)/v
        t = np.linspace(0, zeta, N, endpoint=False)
        #t2 = np.linspace(1, np.exp(0.1*zeta), N, endpoint=False)
        #t = np.concatenate((t1,zeta*0.9+np.log(t2)))

        self.grid = t
        self.f = np.zeros(len(t))
        inte = np.zeros(len(t))

        for i in range(len(self.grid)):
            y = x + v*t[i]
            xd = np.amin([x, -v/2/alpha])
            #print("\n difference of xd", x-xd)
            xu = np.amin([x, v/2/alpha])
            #print("\n difference of xu", x-xu)
            yd = np.amin([y, -v/2/alpha])
            #print("\n difference of xd", y-yd)
            yu = np.amin([y, v/2/alpha])
            #print("\n difference of yu", y-yu)
            inte[i] = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
            #Temp = alpha*t[i]+2*L*alpha/v*np.log((L-x-v*t[i])/(L-x))
            #print("\n difference of proba at ", t[i], -alpha/v*inte[i]-Temp)
            self.f[i] = 1-np.exp(-alpha/v*inte[i])
      

    def plot(self):
        fig = plt.figure()
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(self.grid, self.f)
        plt.draw()



def DistMini(v, x, alpha, L, K=15):
    """Simulate a random variable of distribution proportional to min(L-x, L+v/alpha-x)/min(L+x, L-v/alpha+x)"""
    if v<0:
        print("\n No valid input")
    else:
        zeta = (L-x)/v
    
    A = lambda t: np.minimum(np.array([x, x, x+v*t, x+v*t, t]), np.array([-v/2/alpha, v/2/alpha, -v/2/alpha, v/2/alpha, t]))    
    F = lambda A: np.exp(-alpha*A[4]-2*alpha/v*(A[2]-A[3])+2*alpha/v*(A[0]-A[1])+np.log(L+v/alpha+A[2])-np.log(L+v/alpha+A[0])-2*L*alpha/v*(np.log(L-A[2])-np.log(L-A[3]))+2*L*alpha*v*(np.log(L-A[0])-np.log(L-A[1]))-np.log(L-A[3])+np.log(L-x-v*A[4])+np.log(L-A[1])-np.log(L-x))
    
    x0 = 0
    x1 = zeta
    error = x1 - x0
    mid = (x0+x1)/2
        
    U = np.random.uniform()
    
    k = 0 
    while k < K and error > 1/10**5:

        if F(A(mid))-U >= 0:
            x0 = mid
        else:
            x1 = mid
        k += 1
        error = x1 - x0
        mid = (x0+x1)/2
    
    return mid   


class SubPathMini:
    """Define a subpath of a neutron"""
    def __init__(self, tstart, xstart, vstart, tfin, alpha, a, b):
        # IF IsLinear is TRUE, the scattering function is of the form (alpha * (b-x)/x-a, alpha*(x-a)/(b-x))
        # Otherwise, it is given by (sfun1, sfun2) 
        
        self.tstart = tstart
     
        self.xstart = xstart
        self.vstart = vstart
        self.tfin = tfin
            # tfin is lifetime of the particle. Initially ends at final time.
        
        self.turn = [[self.tstart, self.xstart, self.vstart]]
           # turn is a list of (time, position, direction) where the particle changes directions, 
           # including the starting and ending points.
        self.scatter(self.tfin, alpha, a, b)
        #self.trim(tfin,a,b)
        #print(" tstart= ", self.tstart)
        #print(" tfin= ", self.tfin, self.where(self.tfin))
        #print("changing points", self.turn)


    def scatter(self, tfin, alpha, a, b):
        # generates the list turn 
        
        temp = self.tstart
        position = self.xstart
        speed = self.vstart
        if speed > 0:
            zeta = (b-position)/speed
            #rate = slist[index]
        else:
            zeta = (a-position)/speed
            #print(speed,zeta)
            #rate = slist[index]
        alive = 1
        while temp < tfin and alive == 1:
            
            if speed > 0:
                Exp = DistMini(speed, position, alpha, b)
            else:
                Exp = DistMini(-speed, -position, alpha, b)


            temp += Exp
            if np.amin([temp,temp-Exp+zeta]) < tfin:
                if Exp >= zeta:
                   alive = 0
                   self.tfin = temp-Exp+zeta
                   #print("hit the boundary",self.tfin, position+speed*zeta,speed)
                   self.turn.append([self.tfin,position+speed*zeta,speed])
                else:
                    position += speed * Exp
                    speed = -speed
                    if speed > 0:
                        zeta = (b-position)/speed
                    else:
                        zeta = (a-position)/speed
                        #print("zeta=", zeta)
                    self.turn.append([temp,position,speed])
        if alive == 1:
            self.turn.append([tfin,self.turn[-1][1]+(tfin-self.turn[-1][0])*self.turn[-1][2],self.turn[-1][2]])
 
        # Add the final point
        return self.turn

    
    def where(self, temps):
        # returns the position and direction of the particle of a given time
        if temps < self.tstart or temps > self.tfin:
            print(self.tstart, self.tfin, temps)
            return [-100, -100]
        if len(self.turn) <= 1:
            return [self.xstart+(temps-self.tstart)*self.vstart, self.vstart]
        i = 0
        while i < len(self.turn)-1:
            if temps >= self.turn[i][0] and temps <= self.turn[i+1][0]:
                return [self.turn[i][1]+(temps-self.turn[i][0]) * self.turn[i][2], self.turn[i][2]]
                break
            else:
                i += 1
        print("\n Oops, something wrong...")
        return [-200, -200]
            
    


class PathsMini:
    """Define a collection of neutron trajectories"""
    def __init__(self, tstart, tfin, xstart, vstart, beta, srate, a, b, max_particles = 500000):

        self.trajectories = []
        self.tstart = tstart
        self.xstart = xstart
        self.vstart = vstart
        self.alpha = srate
        self.beta = beta
        self.a = a
        self.b = b

    
        self.trajectories.append(SubPathMini(tstart, xstart, vstart, tfin, srate, a, b))
    
                    
    def plot(self):
        # %matplotlib inline

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        
        color = iter(plt.cm.rainbow(np.linspace(0,1,len(self.trajectories))))
        
        for x in self.trajectories:
            c = next(color)
            t_local = [row[0] for row in  x.turn]
            traj = []
            for ts in t_local:
                traj.append([x.where(ts)[0]])
                #print("for plotting", ts, x.where(ts)[0])
            axes.plot(t_local, traj, c = c)

        plt.draw()
    
    def integral(self, time_start, time):
        # Compute the weight integral. Only works for h_+=b-x and h_-=x-a
        L = self.b
        alpha = self.alpha
        I = -self.alpha*(time-time_start)
        #Ic = -self.alpha*(time-time_start)
        #Icc = -2*self.alpha*(time-time_start)
        #temp = I
        #TTp = 0
        #print("\n starting and ending times", time_start, time)
        v = np.absolute(self.vstart)
        for tr in self.trajectories:
            i = 0
            while i < len(tr.turn)-1 and tr.turn[i+1][0] < time:
                s = tr.turn[i]
                ss = tr.turn[i+1]
                if s[0] > time_start:
                   if s[2] > 0:
                      I += np.log(np.amin([L-ss[1], L+ss[1]+v/alpha]))-np.log(np.amin([L-s[1], L+s[1]+v/alpha]))
                      
                      x = s[1]
                      y = ss[1]
                      xd = np.amin([x, -v/2/alpha])
                      xu = np.amin([x, v/2/alpha])
                      yd = np.amin([y, -v/2/alpha])
                      yu = np.amin([y, v/2/alpha])
                      inte = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
                      I += self.alpha/v*inte
                      
                      #Ic += -alpha*(ss[0]-s[0])+(self.alpha*(self.b-self.a)/v-1)*np.log((self.b-s[1])/(self.b-ss[1]))
                      #Icc += (self.alpha*(self.b-self.a)/v-1)*np.log((self.b-s[1])/(self.b-ss[1]))
                      #TTp += -alpha*(ss[0]-s[0])
                      #print("\n term", s[0], ss[0])
                      #print("\n Case 1:difference of integral", I-Ic)
                   
                   else:
                      I += np.log(np.amin([L+ss[1], L-ss[1]+v/alpha]))-np.log(np.amin([L+s[1], L-s[1]+v/alpha]))
                      
                      x = -s[1]
                      y = -ss[1]
                      xd = np.amin([x, -v/2/alpha])
                      xu = np.amin([x, v/2/alpha])
                      yd = np.amin([y, -v/2/alpha])
                      yu = np.amin([y, v/2/alpha])
                      inte = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
                      I += self.alpha/v*inte
                      
                      #Ic += -alpha*(ss[0]-s[0])+(self.alpha*(self.b-self.a)/v-1)*np.log((s[1]-self.a)/(ss[1]-self.a))
                      #Icc += (self.alpha*(self.b-self.a)/v-1)*np.log((s[1]-self.a)/(ss[1]-self.a))
                      #TTp += -alpha*(ss[0]-s[0])
                      #print("\n term", s[0], ss[0])
                      #print("\n Case 2:difference of integral", I-Ic)
                      
                elif s[0] <= time_start and ss[0] > time_start:
                    if s[2] > 0:
                        g = tr.where(time_start)[0]
                        
                        I += np.log(np.amin([L-ss[1], L+ss[1]+v/alpha]))-np.log(np.amin([L-g, L+g+v/alpha]))
                        x = g
                        y = ss[1]
                        xd = np.amin([x, -v/2/alpha])
                        xu = np.amin([x, v/2/alpha])
                        yd = np.amin([y, -v/2/alpha])
                        yu = np.amin([y, v/2/alpha])
                        inte = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
                        
                        I += self.alpha/v*inte
                        
                        #Ic += -alpha*(ss[0]-time_start)+(self.alpha*(self.b-self.a)/v-1)*np.log((self.b-tr.where(time_start)[0])/(self.b-ss[1]))
                        #Icc += (self.alpha*(self.b-self.a)/v-1)*np.log((self.b-tr.where(time_start)[0])/(self.b-ss[1]))
                        #TTp += -alpha*(ss[0]-time_start)
                        #print("\n term", time_start, ss[0])
                        #print("\n Case 3:difference of integral", I-Ic)
                        
                    else :
                        g = tr.where(time_start)[0]
                        I += np.log(np.amin([L+ss[1], L-ss[1]+v/alpha]))-np.log(np.amin([L+g, L-g+v/alpha]))
                        
                        x = -g
                        y = -ss[1]
                        xd = np.amin([x, -v/2/alpha])
                        xu = np.amin([x, v/2/alpha])
                        yd = np.amin([y, -v/2/alpha])
                        yu = np.amin([y, v/2/alpha])
                        inte = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
                        
                        I += self.alpha/v*inte
                        
                        #Ic += -alpha*(ss[0]-time_start)+(self.alpha*(self.b-self.a)/v-1)*np.log((tr.where(time_start)[0]-self.a)/(ss[1]-self.a))
                        #Icc += (self.alpha*(self.b-self.a)/v-1)*np.log((tr.where(time_start)[0]-self.a)/(ss[1]-self.a))
                        #TTp += -alpha*(ss[0]-time_start)
                        #print("\n term", time_start, ss[0])
                        #print("\n Case 4:difference of integral", I-Ic)
                        
                i += 1
                
            if tr.turn[i][0] < time :
                if tr.turn[i][2] > 0:
                      h = tr.where(time)[0]
                      g = tr.turn[i][1]
                      I += np.log(np.amin([L-h, L+h+v/alpha]))-np.log(np.amin([L-g, L+g+v/alpha]))
                      
                      x = g
                      y = h
                      xd = np.amin([x, -v/2/alpha])
                      xu = np.amin([x, v/2/alpha])
                      yd = np.amin([y, -v/2/alpha])
                      yu = np.amin([y, v/2/alpha])
                      inte = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
                        
                      I += self.alpha/v*inte
                      
                      #Ic += -alpha*(time-tr.turn[i][0])+(self.alpha*(self.b-self.a)/v-1)*np.log((self.b-tr.turn[i][1])/(self.b-tr.where(time)[0]))
                      #Icc += (self.alpha*(self.b-self.a)/v-1)*np.log((self.b-tr.turn[i][1])/(self.b-tr.where(time)[0]))
                      #TTp += -alpha*(time-tr.turn[i][0])
                      #print("\n term", tr.turn[i][0], time)
                      #print("\n Case 5:difference of integral", I-Ic)
                      
                else: 
                      h = tr.where(time)[0]
                      g = tr.turn[i][1]
                      I += np.log(np.amin([L+h, L-h+v/alpha]))-np.log(np.amin([L+g, L-g+v/alpha]))
                      
                      x = -g
                      y = -h
                      xd = np.amin([x, -v/2/alpha])
                      xu = np.amin([x, v/2/alpha])
                      yd = np.amin([y, -v/2/alpha])
                      yu = np.amin([y, v/2/alpha])
                      inte = 2*(yd-yu)-2*(xd-xu)+y-x-v/alpha*(np.log(L+yd+v/alpha)-np.log(L+v/alpha+xd))+2*L*(np.log(L-yd)-np.log(L-yu))-2*L*(np.log(L-xd)-np.log(L-xu))+v/alpha*(np.log(L-yu)-np.log(L-y))-v/alpha*(np.log(L-xu)-np.log(L-x))
                        
                      I += self.alpha/v*inte
                      
                      #Ic += -alpha*(time-tr.turn[i][0])+(self.alpha*(self.b-self.a)/v-1)*np.log((tr.turn[i][1]-self.a)/(tr.where(time)[0]-self.a))
                      #Icc += (self.alpha*(self.b-self.a)/v-1)*np.log((tr.turn[i][1]-self.a)/(tr.where(time)[0]-self.a))
                      #TTp += -alpha*(time-tr.turn[i][0])
                      #print("\n term", tr.turn[i][0], time)
                      #print("\n Case 6:difference of integral", I-Ic)
        
        #print("\n Final:difference of integral", Ic-Icc, temp-TTp)
        #return np.exp(Ic)              
        return np.exp(I)
    
    
    
    
class DistProd:
    """Compute the distribution function of a quadratic rate fucntion alpha(L-u)(L+u+v/alpha)"""
    def __init__(self, v, x, alpha, L, N=500):
        # v is the current speed of the particle and x its position. 
        if v < 0:
            print("\n No valid input")
        else:
            zeta = (L-x)/v
        t = np.linspace(0, zeta, N, endpoint=False)
        #t2 = np.linspace(1, np.exp(0.1*zeta), N, endpoint=False)
        #t = np.concatenate((t1,zeta*0.9+np.log(t2)))

        self.grid = t
        self.f = np.zeros(len(t))
        inte = np.zeros(len(t))
        
        for i in range(len(self.grid)):
            y = x+ v*t[i]
            inte[i] = y-x + 2*v/(2*L*alpha+v)*(L*(np.log(L-x)-np.log(L-y))-(L+v/alpha)*(np.log(L+y+v/alpha)-np.log(L+x+v/alpha)))
            self.f[i] = 1-np.exp(-alpha/v*inte[i])

    def plot(self):
        fig = plt.figure()
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(self.grid, self.f)
        plt.draw()

        
def Dist_Prod(v, x, alpha, L, K=15):
    """Simulate a random variable of distribution proportional to (L-x)(L+v/alpha-x)/(L+x)(L-v/alpha+x)"""
    if v<0:
        print("\n No valid input")
    else:
        zeta = (L-x)/v
            
    F = lambda t: np.exp(-alpha*t-2*alpha/(2*L*alpha+v)*(L*(np.log(L-x)-np.log(L-x-v*t))-(L+v/alpha)*(np.log(L+x+v*t+v/alpha)-np.log(L+x+v/alpha))))
    
    x0 = 0
    x1 = zeta
    error = x1 - x0
    mid = (x0+x1)/2
        
    U = np.random.uniform()
    
    k = 0 
    while k < K and error > 1/10**5:

        if F(mid)-U >= 0:
            x0 = mid
        else:
            x1 = mid
        k += 1
        error = x1 - x0
        mid = (x0+x1)/2
    
    return mid




class SubPathProd:
    """Define a subpath of a neutron"""
    def __init__(self, tstart, xstart, vstart, tfin, alpha, a, b):
        # IF IsLinear is TRUE, the scattering function is of the form (alpha * (b-x)/x-a, alpha*(x-a)/(b-x))
        # Otherwise, it is given by (sfun1, sfun2) 
        
        self.tstart = tstart
     
        self.xstart = xstart
        self.vstart = vstart
        self.tfin = tfin
            # tfin is lifetime of the particle. Initially ends at final time.
        
        self.turn = [[self.tstart, self.xstart, self.vstart]]
           # turn is a list of (time, position, direction) where the particle changes directions, 
           # including the starting and ending points.
        self.scatter(self.tfin, alpha, a, b)
        #self.trim(tfin,a,b)
        #print(" tstart= ", self.tstart)
        #print(" tfin= ", self.tfin, self.where(self.tfin))
        #print("changing points", self.turn)


    def scatter(self, tfin, alpha, a, b):
        # generates the list turn 
        
        temp = self.tstart
        position = self.xstart
        speed = self.vstart
        if speed > 0:
            zeta = (b-position)/speed
            #rate = slist[index]
        else:
            zeta = (a-position)/speed
            #print(speed,zeta)
            #rate = slist[index]
        alive = 1
        while temp < tfin and alive == 1:
            
            if speed > 0:
                Exp = Dist_Prod(speed, position, alpha, b)
            else:
                Exp = Dist_Prod(-speed, -position, alpha, b)
            #DistF(speed, position, alpha, a, b).plot()
            
            
            temp += Exp
            if np.amin([temp,temp-Exp+zeta]) < tfin:
                if Exp >= zeta:
                   alive = 0
                   self.tfin = temp-Exp+zeta
                   print("hit the boundary")
                   self.turn.append([self.tfin,position+speed*zeta,speed])
                else:
                    position += speed * Exp
                    speed = -speed
                    if speed > 0:
                        zeta = (b-position)/speed
                    else:
                        zeta = (a-position)/speed
                        #print("zeta=", zeta)
                    self.turn.append([temp,position,speed])
        if alive == 1:
            self.turn.append([tfin,self.turn[-1][1]+(tfin-self.turn[-1][0])*self.turn[-1][2],self.turn[-1][2]])
            #print("last point",self.turn[-1])
            #Need to do something--done
        # Add the final point
        return self.turn

    
    def where(self, temps):
        # returns the position and direction of the particle of a given time
        if temps < self.tstart or temps > self.tfin:
            print(self.tstart, self.tfin, temps)
            #Some tfin < tstart, find out why!!
            return [-100, -100]
        if len(self.turn) <= 1:
            return [self.xstart+(temps-self.tstart)*self.vstart, self.vstart]
        i = 0
        while i < len(self.turn)-1:
            if temps >= self.turn[i][0] and temps <= self.turn[i+1][0]:
                return [self.turn[i][1]+(temps-self.turn[i][0]) * self.turn[i][2], self.turn[i][2]]
                break
            else:
                i += 1
        print("\n Oops, something wrong...")
        return [-200, -200]
      
            


class PathsProd:
    """Define a collection of neutron trajectories"""
    def __init__(self, tstart, tfin, xstart, vstart, beta, srate, a, b):

        self.trajectories = []
        self.tstart = tstart
        self.xstart = xstart
        self.vstart = vstart
        self.alpha = srate
        self.beta = beta
        self.a = a
        self.b = b

    
        self.trajectories.append(SubPathProd(tstart, xstart, vstart, tfin, srate, a, b))
    
                    
    def plot(self):
        # %matplotlib inline

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        
        color = iter(plt.cm.rainbow(np.linspace(0,1,len(self.trajectories))))
        
        for x in self.trajectories:
            c = next(color)
            t_local = [row[0] for row in  x.turn]
            traj = []
            for ts in t_local:
                traj.append([x.where(ts)[0]])
                #print("for plotting", ts, x.where(ts)[0])
            axes.plot(t_local, traj, c = c)

        plt.draw()
    
    def integral(self, time_start, time):
        # Compute the weight integral. Only works for h_+=b-x and h_-=x-a
        L = self.b
        alpha = self.alpha
        I = -self.alpha*(time-time_start)
        #I = 0
        v = np.absolute(self.vstart)
        for x in self.trajectories:
            i = 0
            while i < len(x.turn)-1 and x.turn[i+1][0] < time:
                s = x.turn[i]
                ss = x.turn[i+1]
                if s[0] > time_start:
                   if s[2] > 0:
                      I += np.log(L-ss[1])+np.log(L+ss[1]+v/alpha)-np.log(L-s[1])-np.log(L+s[1]+v/alpha)
                      inte = ss[1]-s[1] + 2*v/(2*L*alpha+v)*(L*np.log(L-s[1])-L*np.log(L-ss[1])-(L+v/alpha)*np.log(L+ss[1]+v/alpha)+(L+v/alpha)*np.log(L+s[1]+v/alpha))
                      I += alpha/v*inte
                 
                   else:
                      I += np.log(L+ss[1])+np.log(L-ss[1]+v/alpha)-np.log(L+s[1])-np.log(L-s[1]+v/alpha)
                      inte = -ss[1]+s[1] + 2*v/(2*L*alpha+v)*(L*(np.log(L+s[1])-np.log(L+ss[1]))-(L+v/alpha)*(np.log(L-ss[1]+v/alpha)-np.log(L-s[1]+v/alpha)))
                      I += alpha/v*inte
                
                elif s[0] <= time_start and ss[0] > time_start:
                    if s[2] > 0:
                      g = x.where(time_start)[0]
                      I += np.log(L-ss[1])+np.log(L+ss[1]+v/alpha)-np.log(L-g)-np.log(L+g+v/alpha)
                      inte = ss[1]-g + 2*v/(2*L*alpha+v)*(L*(np.log(L-g)-np.log(L-ss[1]))-(L+v/alpha)*(np.log(L+ss[1]+v/alpha)-np.log(L+g+v/alpha)))
                      I += alpha/v*inte

                    else :
                      g = -x.where(time_start)[0]
                      I += np.log(L+ss[1])+np.log(L-ss[1]+v/alpha)-np.log(L-g)-np.log(L+g+v/alpha)
                      inte = -ss[1]-g + 2*v/(2*L*alpha+v)*(L*(np.log(L-g)-np.log(L+ss[1]))-(L+v/alpha)*(np.log(L-ss[1]+v/alpha)-np.log(L+g+v/alpha)))
                      I += alpha/v*inte

                i += 1
                
            if x.turn[i][0] < time :
                if x.turn[i][2] > 0:
                   g = x.turn[i][1]
                   h = x.where(time)[0]
                   I += np.log(L-h)+np.log(L+h+v/alpha)-np.log(L-g)-np.log(L+g+v/alpha)
                   inte = h-g + 2*v/(2*L*alpha+v)*(L*(np.log(L-g)-np.log(L-h))-(L+v/alpha)*(np.log(L+h+v/alpha)-np.log(L+g+v/alpha)))
                   I += alpha/v*inte
                else: 
                   g = -x.turn[i][1]
                   h = -x.where(time)[0]
                   I += np.log(L-h)+np.log(L+h+v/alpha)-np.log(L-g)-np.log(L+g+v/alpha)
                   inte = h-g + 2*v/(2*L*alpha+v)*(L*(np.log(L-g)-np.log(L-h))-(L+v/alpha)*(np.log(L+h+v/alpha)-np.log(L+g+v/alpha)))
                   I += alpha/v*inte
        return np.exp(I)
