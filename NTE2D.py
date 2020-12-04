"""
Implement a Vanilla NBP and NRW model for MC simulation of the NTE.

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
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter


class circle:
    """Shape to determine area of scatter/branching."""

    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius

    def time_in_circle(self, pos, theta, v):
        """Compute entry, exit times in circle for given trajectory."""
        a = v**2
        b = 2*v*((pos[0] - self.centre[0])*np.cos(theta)
                 + (pos[1] - self.centre[1])*np.sin(theta))
        c = ((pos[0] - self.centre[0])**2 + (pos[1] - self.centre[1])**2
             - self.radius**2)
        det = b**2 - 4*a*c
        if det < 0:
            self.entry_time = -100
            self.exit_time = -100
        elif det == 0:
            self.entry_time = (-b - np.sqrt(det))/(2*a)
            self.exit_time = self.entry_time
        else:
            if -b - np.sqrt(det) < 0:
                if -b + np.sqrt(det) > 0:
                    self.entry_time = 0
                    self.exit_time = (-b + np.sqrt(det))/(2*a)
                else:
                    self.entry_time = -100
                    self.exit_time = -100
            else:
                self.entry_time = (-b - np.sqrt(det))/(2*a)
                self.exit_time = (-b + np.sqrt(det))/(2*a)
        return self.entry_time, self.exit_time

    def interval_in_circle(self, traj, max_time):
        """Compute entry, exit times for a given "stick"."""
        en, ex = self.time_in_circle(traj[1], traj[2], traj[3])
        if (en >= 0 and en < (max_time-traj[0])):
            return traj[0]+en, min(traj[0]+ex, max_time)
        else:
            return None

    def in_circle(self, pos):
        """Return true if the position is in the circle."""
        return ((pos[0]-self.centre[0])**2. + (pos[1]-self.centre[1])**2.
                <= self.radius**2.)


class domain:
    """Define a spatial domain (in 2d) with Scattering rates and boundaries."""

    def __init__(self, L=1, A=np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]),
                 b=np.array([1.0, 1.0, 1.0, 1.0])):
        # L is outer box size. Might be used to discretize space in e.g.
        #   plotting.
        # A, b define the domain, so x is in the domain if and only if
        # |x_i| <= L and A x \le b for each coordinate.
        # The default is the box [-1,1] x [-1,1]

        # List of scatter locations
        self.A = A
        self.b = b
        self.L = L

    def is_in_domain(self, pos):
        """Is the position pos in the Domain."""
        return np.all((self.b - self.A@pos >= 0))

    def exit_time(self, pos, theta, v=1.0):
        """When does the particle exit the Domain."""
        z = np.array([np.cos(theta), np.sin(theta)])
        return min(filter(lambda x: x >= 0, (self.b-self.A@pos)/(v*self.A@z)))

    def exit_time_reverse(self, pos, theta, v=1.0):
        """When does the particle exit the domain, travelling in reverse."""
        z = np.array([np.cos(theta), np.sin(theta)])
        return min(filter(lambda x: x >= 0,
                          (self.A@pos - self.b)/(v*self.A@z)))


class SubPathCircles:
    """Define a subpath of a 2D-neutron trajectory through domain with circles.

    Note that in circle, rates are for additional branching/scattering
       over base rate of domain.
    """

    def __init__(self, tstart, xstart, thetastart, vstart, tfin, srate_dom,
                 srate_circ, brate_dom, brate_circ):
        self.tstart = tstart
        self.xstart = xstart
        self.thetastart = thetastart
        self.vstart = vstart
        self.tfin = tfin
        self.drate = srate_dom
        self.crate = srate_circ
        self.drate_b = brate_dom
        self.crate_b = brate_circ
        self.alive = True

        self.c1 = circle(np.array([0.5, 0.5]), 0.25)
        self.c2 = circle(np.array([0.5, -0.5]), 0.25)
        self.c3 = circle(np.array([-0.5, 0.5]), 0.25)
        self.c4 = circle(np.array([-0.5, -0.5]), 0.25)
        self.c = [self.c1, self.c2, self.c3, self.c4]
        self.d = domain()
        self.circle_time = []

        self.turn = [[self.tstart, self.xstart, self.thetastart, self.vstart]]

        self.scatter(self.tfin, self.drate, self.crate, self.drate_b,
                     self.crate_b)
        # self.time_in_circle(self.c)

    def in_circle(self, time_in, time_out, event_time):
        """Return true if particle is in any circle at event time.

        Expects time_in, time_out to be entry/exit times of circle. Event
        time is time after the entry.
        """
        return (time_in + event_time <= time_out)

    def scatter(self, tfin, srate_dom, srate_circ, brate_dom, brate_circ):
        """
        Generate paths up to time tfin.

        Parameters
        ----------
        tfin : TYPE real
            Time to generate paths up to.
        srate_dom : TYPE
            Scatter rate in the domain.
        srate_circ : TYPE
            Extra scatter rate observed in circles.
        brate_dom : TYPE
            Branch rate in domain.
        brate_circ : TYPE
            Extra scatter observed in circles.

        Returns
        -------
        TYPE list
            List of times and places of scatter events up to tfin, or death.

            If particle survives, last location is not a scatter.

        """
        zeta = self.d.exit_time(self.xstart, self.thetastart, self.vstart)
        temp = self.tstart
        position = self.xstart
        angle = self.thetastart
        speed = self.vstart

        while temp < tfin and self.alive:
            circle_times = [circ.time_in_circle(position, angle, speed)
                            for circ in self.c]
            if (srate_dom+brate_dom) > 0:
                d_exp = -np.log(np.random.uniform())/(srate_dom+brate_dom)
            else:
                d_exp = math.inf
            if (srate_circ+brate_circ) > 0:
                c_exp = (-np.log(np.random.uniform(size=len(self.c))) /
                         (srate_circ + brate_circ)).tolist()

                Exp = min(d_exp,
                          min([math.inf] +
                              [(circle_times[c_exp.index(stime)][0]
                                + stime)
                               for stime in c_exp if
                               self.in_circle(
                                   circle_times[c_exp.index(stime)][0],
                                   circle_times[c_exp.index(stime)][1],
                                   stime)]))

            else:
                Exp = d_exp

            # if (Exp == math.inf) and (temp + zeta < tfin):
            #     self.alive = False
            #     self.tfin = temp + zeta
            #     self.turn.append([self.tfin,
            #                       [position[0]+speed*np.cos(angle)*zeta,
            #                        position[1]+speed*np.sin(angle)*zeta],
            #                       angle, speed])

            # temp += Exp
            if np.min([temp + Exp, temp + zeta]) < tfin:
                if Exp >= zeta:
                    # Hits Boundary before tfin
                    self.alive = False
                    self.tfin = temp + zeta
                    self.turn.append([temp + zeta,
                                      [position[0] +
                                       speed * np.cos(angle) * zeta,
                                       position[1] +
                                       speed * np.sin(angle) * zeta],
                                      angle, speed])
                    temp = temp + zeta
                else:
                    # Scatter before tfin
                    position = [position[0] + np.cos(angle) * speed * Exp,
                                position[1] + np.sin(angle) * speed * Exp]
                    angle = np.random.uniform(0, 2 * np.pi)-np.pi
                    speed = self.vstart
                    zeta = self.d.exit_time(position, angle, speed)
                    self.tfin = temp + Exp
                    temp = temp + Exp

                    self.turn.append([temp, position, angle, speed])
            else:
                # reach tfin:
                position = [position[0] + np.cos(angle) * speed *
                            (tfin-temp),
                            position[1] + np.sin(angle) * speed *
                            (tfin-temp)]
                self.tfin = tfin
                self.turn.append([tfin, position, angle, speed])
                temp = tfin

        # if self.alive:
        #     self.turn.append([tfin, [self.turn[-1][1][0]+(tfin-self.turn[-1][0])*self.turn[-1][3]*np.cos(self.turn[-1][2]), self.turn[-1][1][1]+(tfin-self.turn[-1][0])*self.turn[-1][3]*np.sin(self.turn[-1][2])], self.turn[-1][2], self.turn[-1][3]])
        return self.turn

    def where(self, temps):
        """Return the position, angle of the particle at a given time."""
        if temps < self.tstart or temps > self.tfin:
            return [[-100, -100], 0, 0]
        if len(self.turn) <= 1:
            return [[self.xstart[0]+(temps-self.tstart)*self.vstart*np.cos(self.thetastart), self.xstart[1]+(temps-self.tstart)*self.vstart*np.sin(self.thetastart)], self.thetastart, self.vstart]
        i = 0
        while i < len(self.turn)-1:
            if temps >= self.turn[i][0] and temps <= self.turn[i+1][0]:
                return [[self.turn[i][1][0]+(temps-self.turn[i][0])*self.turn[i][3]*np.cos(self.turn[i][2]), self.turn[i][1][1]+(temps-self.turn[i][0])*self.turn[i][3]*np.sin(self.turn[i][2])], self.turn[i][2], self.turn[i][3]]
                break
            else:
                i += 1
        return [[-200, -200], 0, 0]

    def time_in_circle(self, circles):
        """Return a list of the entry and exit times from circles."""
        i = 0
        circle_time = []
        while i < len(self.turn):
            # Last entry in self.turn can either be the particle position at
            #   tfin, or the exit time from the domain, if killed.
            x = self.turn[i]
            if i < len(self.turn)-1:
                mx_time = self.turn[i+1][0]
            else:
                mx_time = np.inf
            for c in circles:
                interval = c.interval_in_circle(x, mx_time)
                if interval:
                    # interval is None if no times.
                    circle_time.append(interval)
            i += 1
        return circle_time

    def dress(self, srate_dom, srate_circ, brate_dom, brate_circ,
              branch_times):
        """Return positions and times for branching, add to branch_times."""
        for turn in self.turn[1:-1]:
            in_circ = np.any([c.in_circle(turn[1]) for c in self.c])
            if in_circ:
                is_scatter = (np.random.uniform() <
                              ((srate_dom + srate_circ) /
                               (srate_dom + brate_dom + srate_circ
                                + brate_circ)))
            else:
                is_scatter = (np.random.uniform() <
                              (srate_dom / (srate_dom + brate_dom)))

            if not is_scatter:
                branch_times.append([turn[0], turn[1],
                                     np.random.uniform(0, 2 * np.pi)
                                     - np.pi, turn[3]])

        return branch_times

    def is_alive(self, time_val):
        """Check if the particle is alive at a given time."""
        return ((self.tstart <= time_val < self.tfin) or
                ((time_val == self.tfin) and self.alive))


class PathCircles:
    """Define a collection of neutron trajectories in 2D."""

    def __init__(self, tstart, tfin, xstart, thetastart, srate_dom, srate_circ,
                 brate_dom, brate_circ, vstart, max_particles=50000):
        self.branch_times = []
        self.trajectories = []
        self.tstart = tstart
        self.xstart = xstart
        self.thetastart = thetastart
        self.vstart = vstart

        self.trajectories.append(SubPathCircles(tstart, xstart, thetastart, vstart, tfin, srate_dom, srate_circ, brate_dom, brate_circ))
        if max(brate_dom, brate_circ) > 0:
            self.trajectories[0].dress(srate_dom, srate_circ, brate_dom, brate_circ, self.branch_times)

            while len(self.trajectories) < max_particles and len(self.branch_times) > 0:
                temps, start, angle, speed = self.branch_times.pop()
                self.trajectories.append(SubPathCircles(temps, start, angle, speed, tfin, srate_dom, srate_circ, brate_dom, brate_circ))
                self.trajectories[-1].dress(srate_dom, srate_circ, brate_dom, brate_circ, self.branch_times)

        if len(self.trajectories) >= max_particles:
            print("\nWARNING: Maximum number of particles exceeded.\n")

    def plot(self):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 2, 2])
        axes.set_aspect('equal')
        axes.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color='black')
        circle1 = plt.Circle((-0.5, -0.5), 0.25, fill=True, facecolor='red')
        circle2 = plt.Circle((-0.5, 0.5), 0.25, fill=True, facecolor='red')
        circle3 = plt.Circle((0.5, -0.5), 0.25, fill=True, facecolor='red')
        circle4 = plt.Circle((0.5, 0.5), 0.25, fill=True, facecolor='red')
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)
        plt.gcf().gca().add_artist(circle4)

        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.trajectories))))

        for z in self.trajectories:
            c = next(color)
            x_traj = []
            y_traj = []
            for point in z.turn:
                x_traj.append(point[1][0])
                y_traj.append(point[1][1])
                axes.plot(x_traj, y_traj, c=c, linewidth=1, marker='o',
                          markersize=3)

        plt.show()

    def count(self, time_val):
        """Count of particles alive at time_val. Can take numpy vec input."""
        if np.isscalar(time_val):
            return sum(1 for x in self.trajectories if x.is_alive(time_val))
        else:
            temp = np.zeros(np.shape(time_val))
            for index in np.ndindex(np.shape(time_val)):
                temp[index] = self.count(time_val[index])
            return temp

    def Integral(self, Brate_dom, Brate_circ, time):
        """Compute path integral int_0^time beta(xi_s)ds for many-to-one."""
        if len(self.trajectories) > 1:
            print("\n Check branching rates. Only implemented for Many-To-One")
        x = self.trajectories[-1]
        if x.tfin < time:
            return 0
        else:
            # circle_time = x.circle_time
            # output = Brate_dom * time
            circ_time = 0
            for I in x.time_in_circle(x.c):
                circ_time += (min(time,I[1])-min(time,I[0]))

            output = (Brate_dom * time +
                      Brate_circ * circ_time)

        return np.exp(output)

    def hmap(self, s):
        """Plot a heatmap of the Particle system."""
        x = []
        y = []
        for z in self.trajectories:
            for point in z.turn[1:-1]:
                x.append(point[1][0])
                y.append(point[1][1])
            if z.alive:
                x.append(z.turn[-1][1][0])
                y.append(z.turn[-1][1][1])

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000)
        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = heatmap.T

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 1.5, 1.5])
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.set_title(r"Smoothing with  $\sigma$ = %d" % s)

        plt.show()

    def hmap2(self, sample_times, v0, v1, s):
        """Plot density of particles passing through each point in the domain.

        Plots points with angle in range (v0, v1), using times in sample_times.
        """
        x = []
        y = []
        for z in self.trajectories:
            for tt in sample_times:
                if abs(z.where(tt)[0][0]) <= 1 and (v0 <=
                                                    z.where(tt)[1] <= v1):
                    x.append(z.where(tt)[0][0])
                    y.append(z.where(tt)[0][1])

        heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000)
        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = heatmap.T

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 1.5, 1.5])
        ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        ax.set_title(r"Smoothing with  $\sigma$ = %d" % s)

        plt.show()

    def finalPositions(self):
        """Return final positions of particles."""
        posit = []
        for p in self.trajectories:
            if p.alive:
                posit.append((*p.turn[-1][1:3], 1.0))
            else:
                posit.append((*p.turn[-1][1:3], 0.0))
        return posit
