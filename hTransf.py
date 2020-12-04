#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
import sys
from operator import itemgetter
from itertools import combinations


class circle:
    """Shape to determine area of scatter/branching."""

    def __init__(self, centre, radius, rates=(0.5, 1.0), offspring=2.0):
        self.centre = centre
        self.radius = radius
        self.scatterRate = rates[0]
        self.branchRate = rates[1]
        self.offspring = offspring
        self.rate = self.scatterRate + (self.offspring-1) * self.branchRate

    def timeInCircle(self, pos, theta, v):
        """Return Entry and Exit time of circle."""
        a = v**2
        b = 2*v*((pos[0] - self.centre[0])*np.cos(theta)
                 + (pos[1] - self.centre[1])*np.sin(theta))
        c = ((pos[0] - self.centre[0])**2 + (pos[1] - self.centre[1])**2
             - self.radius**2)
        det = b**2 - 4*a*c
        if det < 0:
            self.entryTime = -100
            self.exitTime = -100
        elif det == 0:
            self.entryTime = (-b - np.sqrt(det))/(2*a)
            self.exitTime = self.entryTime
        else:
            if -b - np.sqrt(det) < 0:
                if -b + np.sqrt(det) > 0:
                    self.entryTime = 0
                    self.exitTime = (-b + np.sqrt(det))/(2*a)
                else:
                    self.entryTime = -100
                    self.exitTime = -100
            else:
                self.entryTime = (-b - np.sqrt(det))/(2*a)
                self.exitTime = (-b + np.sqrt(det))/(2*a)
        return self.entryTime, self.exitTime

    def inCircle(self, pos):
        """Return True if particle in circle."""
        return ((pos[0]-self.centre[0])**2. + (pos[1]-self.centre[1])**2.
                <= self.radius**2.)


class domain:
    """Define a spatial domain (in 2d) with Scattering rates and boundaries."""

    def __init__(self, L=1, A=np.array([[-1, 0], [1, 0], [0, 1], [0, -1]]),
                 b=np.array([1.0, 1.0, 1.0, 1.0]), centre=(0., 0.),
                 baseRates=(1.0, 0.1),
                 circles=[((0.5, 0.5), 0.25, (0.0, 1.0)),
                          ((-0.5, 0.5), 0.25, (0.0, 1.0)),
                          ((-0.5, -0.5), 0.25, (0.0, 1.0)),
                          ((0.5, -0.5), 0.25, (0.0, 1.0))],
                 offspring=2.0):
        # L is outer box size. Might be used to discretize space in e.g.
        #   plotting.
        # A, b define the domain, so x is in the domain if and only if
        # |x_i| <= L and A x \le b for each coordinate.
        # The default is the box [-1,1] x [-1,1].
        # Offspring is mean number of particles produced at each fission event,
        #  default is binary branching (offspring = 2.)

        self.A = A
        self.b = b
        self.L = L
        self.edges = np.size(b)
        self.centre = centre
        self.scatterRate = baseRates[0]
        self.branchRate = baseRates[1]
        self.offspring = offspring
        self.rate = self.scatterRate + (self.offspring-1.) * self.branchRate

        if isinstance(circles[0], circle):
            self.circles = circles
        else:
            self.circles = [circle(*c, offspring=self.offspring)
                            for c in circles]

        self.corners = []

        # Determine all the corners of the domain:
        for i in range(self.edges-1):
            for j in range(self.edges-i-1):
                if np.linalg.cond(self.A[(i, i+j+1),
                                         :]) < 1/sys.float_info.epsilon:
                    temp = np.linalg.solve(self.A[(i, i+j+1), :],
                                           self.b[np.array((i, i+j+1))])
                    if self.is_in_domain(temp):
                        self.corners.append(temp)

        # Rewrite corners in anti-clockwise order.
        if not self.is_in_domain(self.centre):
            print("Warning: (0., 0.) should be in the domain.")
        else:
            temp = [(p, self.angleFromCentre(p)) for p in self.corners]
            temp = sorted(temp, key=itemgetter(1))
        self.corners = [p[0] for p in temp]

    def angleFromCentre(self, pos):
        """Compute the angle of current position from centre of domain."""
        x = pos[0] - self.centre[0]
        y = pos[1] - self.centre[1]
        temp = (np.arctan(y/x)
                + np.pi*(x < 0) - 2.*np.pi*(x < 0 and y < 0))
        return temp

    def anglesToCorners(self, pos):
        """Return a list of angles from pos to corners of domain."""
        temp = []
        for c in self.corners:
            x = c[0] - pos[0]
            y = c[1] - pos[1]
            temp.append(np.arctan(y/x)
                        + np.pi*(x < 0) - 2.*np.pi*(x < 0 and y < 0))
        return temp

    def is_in_domain(self, pos):
        """Is the position pos in the Domain."""
        return np.all((self.b - self.A@pos >= 0))

    def exitTime(self, pos, theta, v=1.0):
        """When does the particle exit the Domain."""
        z = np.array([np.cos(theta), np.sin(theta)])
        with np.errstate(divide='ignore'):
            # Division by zero as acceptable here
            temp = (self.b-self.A@pos)/(v*self.A@z)
            # if not filter(lambda x: x >= 0, temp):
            #     print(self.b, self.A@pos, pos, self.A@z, temp)

            try:
                temp2 = min(filter(lambda x: x >= 0, temp))
            except:
                print("Error in exitTime\n")
                print(self.b, self.A@pos, pos, z, theta, self.A@z, temp)
                temp2 = 200.
        return temp2

    def exitTimeReverse(self, pos, theta, v=1.0):
        """When does the particle exit the domain, travelling in reverse."""
        z = np.array([np.cos(theta), np.sin(theta)])
        with np.errstate(divide='ignore'):
            temp = (self.A@pos - self.b)/(v*self.A@z)
            try:
                temp2 = min(filter(lambda x: x >= 0, temp))
            except:
                print("\nError in exitTimereverse")
                print(self.b, self.A@pos, pos, z, theta, self.A@z, temp)
                temp2 = 200.
        return temp2

        return min(filter(lambda x: x >= 0, temp))

    def alpha(self, pos):
        """Branching /scatter rate at pos."""
        return (sum([c.rate for c in self.circles if c.inCircle(pos)])
                + self.rate)

    def beta(self, pos):
        """Growth rate at pos."""
        return (sum([c.branchRate * (c.offspring - 1.0) for c in self.circles
                     if c.inCircle(pos)])
                + self.branchRate * (self.offspring - 1.0))

    def maxRate(self, pos1, pos2):
        """Return the maximum rate of events on the path from pos1 to pos2."""
        if pos2[0]==pos1[0]:
            theta = np.pi/2.
        else:
            theta = np.arctan((pos2[1]-pos1[1])/(pos2[0]-pos1[0]))
        dist = np.sqrt((pos2[1]-pos1[1])**2. + (pos2[0]-pos1[0])**2.)
        maxRate = self.rate
        for c in self.circles:
            t2 = c.timeInCircle(pos1, theta, 1.0)
            if (t2[1] >= 0. and t2[0] <= dist):
                maxRate += c.scatterRate + (self.offspring - 1.) * c.branchRate
        return maxRate


class hFunction:
    """Define a function h for performing an h-transform.

    Should include a domain and values c, C, slope which determine the boundary
    values on the reverse and forward faces, and the slope of the "reverse"
    boundary. (The forward boundary has slope 1).
    """

    def __init__(self, d=domain(), c=0.01, C=1.0, slope = 1.0):
        self.d = d
        self.c = c
        self.C = C
        self.slope = slope

        self.hMax = self.c + max([self.dist(x, y) for x, y in
                                  combinations(self.d.corners, 2)])

    def dist(self, x, y):
        """Distance between x and y."""
        return np.sqrt((x[0]-y[0])**2.+(x[1]-y[1])**2.)

    def val(self, pos, theta, v=1.0):
        """Return value of h at pos, theta."""
        if np.isnan(theta):
            print("Error in val", pos, theta)
            raise ValueError("Error in val")
        return min(self.c + self.d.exitTime(pos, theta),
                   self.C + self.slope * self.d.exitTimeReverse(pos, theta))

    def intCircum(self, pos, theta, v=1.0):
        """Return value of int h over theta at pos, theta."""
        # For now, implement a simple numerical integration
        def f(theta0):
            return self.val(pos, theta0, v)

        # Exact method using quad. Is much slower.
        # t1 = time.time()
        # integ = integrate.quad(f, 0., 2.*np.pi, epsrel=1e-3)[0]/(2.*np.pi)
        # print(time.time()-t1)

        # Inexact method using simple numerical integration. Much faster.
        N = 20
        # t1 = time.time()
        integ2 = np.sum(np.vectorize(f)(np.linspace(0, 2*np.pi, N)[:-1]))/(N-1)
        # print(time.time()-t1)
        # print(integ, integ2)
        return integ2

    def J(self, pos, theta, v=1.0):
        """Return value of Jh at pos, theta."""
        return ((self.intCircum(pos, theta, v)-self.val(pos, theta, v))
                * self.d.alpha(pos))

    def alphaH(self, pos, theta, v=1.0):
        """Return value of alpha_h at pos, theta."""
        return (self.intCircum(pos, theta, v)/self.val(pos, theta, v)
                * self.d.alpha(pos))

    def intJh(self, pos1, pos2, v=1.0):
        """Compute the integral from pos1 to pos2 of Jh/h."""
        print("Not Implemented")
        return 0.

    def intBeta(self, pos1, pos2, v=1.0):
        """Compute the integral from pos1 to pos2 of beta."""
        print("Not Implemented")
        return 0.

    def intJhBeta(self, pos1, pos2, v=1.0):
        """Compute the integral from pos1 to pos2 of Jh/h + beta."""
        dist = self.dist(pos1, pos2)
        if dist < 1e-8:
            # print("intJhBeta identical input")
            return 0.
        NPerD = 25  # Number of integration points per distance travelled
        NMin = 10  # Minimum number of integration points
        N = max(NMin, np.ceil(dist * NPerD).astype('int'))
        if (pos2[0] >= pos1[0]):
            #  arctan gives correct angle - in NE or SE corners
            theta = np.arctan((pos2[1] - pos1[1])/(pos2[0] - pos1[0]))
        elif (pos2[1] >= pos1[1]):
            #  Correct from SE to NW corner
            theta = np.arctan((pos2[1] - pos1[1])/(pos2[0] - pos1[0])) + np.pi
        else:
            #  Correct from NE to SW corner
            theta = np.arctan((pos2[1] - pos1[1])/(pos2[0] - pos1[0])) - np.pi

        if np.isnan(theta):
            print("Error in intJhBeta", pos1, pos2, theta)
            raise ValueError("Error in intJhBeta")

        x = np.linspace(pos1[0], pos2[0], N + 1)
        y = np.linspace(pos1[1], pos2[1], N + 1)
        temp = 0.0
        for i in range(N-1):
            temp += (self.J((x[i+1], y[i+1]), theta, v)
                     / self.val((x[i+1], y[i+1]), theta, v))
            temp += self.d.beta((x[i+1], y[i+1]))
        temp += 0.5 * ((self.J((x[0], y[0]), theta, v)
                        / self.val((x[0], y[0]), theta, v))
                       + (self.J((x[N], y[N]), theta, v)
                          / self.val((x[N], y[N]), theta, v)))
        temp += 0.5 * (self.d.beta((x[0], y[0])) + self.d.beta((x[N], y[N])))
        return temp / N * dist / v

    def nextEvent(self, pos, theta, v=1.0, tMax=np.inf):
        """Find next Event when travelling in direction theta from pos."""
        # Next step: implement weight along trajectory.

        foundEvent = False
        currPos = pos
        tries = 0
        successes = 0
        eps = 10e-8 * self.d.L  # Extra move to go beyond edges of regions.
        logWt = 0.
        time = 0.

        # Record how we finish:
        #    0 = Error.
        #    1 = New Scatter.
        #    2 = Hit Boundary.
        #    3 = Reached tMax
        eventType = 0
        foundTheta = False

        if np.isnan(theta):
            print("Error in nextEvent", pos, theta)

        while not foundEvent:
            a = self.d.exitTime(currPos, theta, v)
            b = self.d.exitTimeReverse(currPos, theta, v)

            # Find next potential event
            # Are we in area of increasing or decreasing h?
            if (self.c + a <= self.C + self.slope * b):
                # Decreasing, linear value of h, potentially to zero:
                nextPos = (currPos[0] + a * v * np.cos(theta),
                           currPos[1] + a * v * np.sin(theta))

                alphaMax = self.d.maxRate(currPos, nextPos)

                t = (self.val(currPos, theta) / v
                     * (1 - np.random.uniform()**(v / (self.hMax * alphaMax))))
                t = min(t, tMax - time)
                if t < a:
                    # next event occurs before hitting boundary
                    nextPos = (currPos[0] + v * t * np.cos(theta),
                               currPos[1] + v * t * np.sin(theta))
                    logWt += self.intJhBeta(currPos, nextPos)
                    time += t
                    if (time >= tMax):
                        # tMax reached before event
                        eventType = 3
                        foundEvent = True
                        foundTheta = True
                        thetaNew = theta
                    else:
                        # Scatter Event happens before hitting boundary or tMax
                        tries += 1
                        test = (self.intCircum(nextPos, theta) / self.hMax
                                * self.d.alpha(nextPos) / alphaMax)
                        #  print(nextPos, theta, test)
                        if (np.random.uniform() <= test):
                            successes += 1
                            eventType = 1
                            foundEvent = True
                else:
                    nextPos = (currPos[0] + v * a * np.cos(theta),
                               currPos[1] + v * a * np.sin(theta))
                    logWt += -np.inf  # Particle left the system
                    time += a
                    foundEvent = True
                    foundTheta = True
                    thetaNew = theta
                    eventType = 2
            else:
                # Increasing value of h, try to get to decreasing:
                critDist = (self.c + a - self.C - self.slope * b)/2.
                nextPos = (currPos[0] + critDist * np.cos(theta),
                           currPos[1] + critDist * np.sin(theta))

                alphaMax = self.d.maxRate(currPos, nextPos)

                t = (self.val(currPos, theta) / v
                     * (np.random.uniform()**(-v /
                                              (self.hMax * alphaMax)) - 1.))
                t = min(t, tMax - time)
                if t >= critDist / v:
                    # Reach decreasing h area before scatter
                    nextPos = (currPos[0] + (critDist + eps) * np.cos(theta),
                               currPos[1] + (critDist + eps) * np.sin(theta))
                    time += critDist / v
                    logWt += self.intJhBeta(currPos, nextPos)
                else:
                    # Potential event before reaching boundary
                    nextPos = (currPos[0] + v * t * np.cos(theta),
                               currPos[1] + v * t * np.sin(theta))
                    logWt += self.intJhBeta(currPos, nextPos)
                    time += t
                    if (time >= tMax):
                        eventType = 3
                        foundEvent = True
                        foundTheta = True
                        thetaNew = theta
                    else:
                        tries += 1
                        test = (self.intCircum(currPos, theta) / self.hMax
                                * self.d.alpha(currPos) / alphaMax)
                        #  print(nextPos, theta, test)
                        if (np.random.uniform() <= test):
                            successes += 1
                            eventType = 1
                            foundEvent = True
            # print(currPos, nextPos, time, tries, successes, logWt)
            currPos = nextPos

        while not foundTheta:
            thetaNew = np.random.uniform(-np.pi, np.pi)
            tries += 1
            if (np.random.uniform()
                    <= self.val(currPos, thetaNew) / self.hMax):
                foundTheta = True
                successes += 1
                logWt += np.log(self.val(currPos, theta) /
                                self.val(currPos, thetaNew))

        return (currPos, thetaNew, v, time, eventType, np.exp(logWt))


class hRW:
    """Define an h-transform with h taken as an Urts function."""

    def __init__(self, tstart, tfin, xstart, thetastart, srate_dom, srate_circ,
                 brate_dom, brate_circ, v, width=1, c=0.01, C=1, N=100,
                 slope=1.):

        self.tstart = tstart
        self.xstart = xstart
        self.thetastart = thetastart
        self.tfin = tfin
        self.drate = srate_dom + brate_dom
        self.crate = srate_circ + brate_circ
        self.srate_dom = srate_dom
        self.srate_circ = srate_circ
        self.brate_dom = brate_dom
        self.brate_circ = brate_circ
        self.const = C
        self.littleConst = c
        self.L = width
        self.vstart = v
        self.alive = True
        self.tCurrent = self.tstart
        self.slope = slope

        self.c1 = circle(np.array([0.5, 0.5]), 0.25, rates=(self.srate_circ,
                                                            self.brate_circ))
        self.c2 = circle(np.array([0.5, -0.5]), 0.25, rates=(self.srate_circ,
                                                             self.brate_circ))
        self.c3 = circle(np.array([-0.5, 0.5]), 0.25, rates=(self.srate_circ,
                                                             self.brate_circ))
        self.c4 = circle(np.array([-0.5, -0.5]), 0.25, rates=(self.srate_circ,
                                                              self.brate_circ))
        self.c = [self.c1, self.c2, self.c3, self.c4]
        self.d = domain(L=self.L, baseRates=(self.srate_dom, self.brate_dom),
                        circles=self.c)

        self.h = hFunction(self.d, c=self.littleConst, C=self.const,
                           slope=self.slope)

        self.counter = []  # records the rejection times

        self.turn = [[self.tstart, self.xstart, self.thetastart, 1.0]]

        self.scatter(self.tfin)

        # self.circle_time = []
        # self.timeInCircle(self.c)

    def scatter(self, tFin):
        """Compute the scattering path and weights up to time tFin."""
        temp = self.tCurrent
        pos = self.xstart
        angle = self.thetastart
        speed = self.vstart
        logWt = 0.0
        # logWt = +np.log(self.h.val(pos, angle))

        while (temp < tFin) and self.alive:

            # finds the next scattering event
            scatter = self.h.nextEvent(pos, angle, speed, tMax=(tFin-temp))
            if np.isnan(scatter[1]):
                print("Theta is nan:", pos, angle, speed, tFin-temp, scatter)
            pos = scatter[0]
            angle = scatter[1]
            speed = scatter[2]
            temp += scatter[3]
            with np.errstate(divide='ignore'):
                # Suppress warning if particle gets killed.
                # logWt += np.log(scatter[5]) - np.log(self.h.val(pos, angle))
                logWt += np.log(scatter[5])

            if scatter[4] == 0:
                print("Error in hTransform: nextEvent")
            elif scatter[4] == 1:
                # New Scatter Event
                self.turn.append([temp, pos, angle, np.exp(logWt)])
            elif scatter[4] == 2:
                # Exit Domain
                self.alive = False
                self.turn.append([temp, pos, angle, 0.])
            elif scatter[4] == 3:
                # Time reaches max
                self.turn.append([temp, pos, angle, np.exp(logWt)])
        self.tCurrent = temp
        return self.turn

    def plot(self, title=""):
        # Should be updated!
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 2, 2])
        axes.set_aspect('equal')
        plt.xlim(-1., 1.)
        plt.ylim(-1., 1.)
        # axes = fig.add_axes([0.1, 0.1, 2, 2])
        # axes.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color = 'black')
        circle1 = plt.Circle((-0.5, -0.5), 0.25, fill=False)
        circle2 = plt.Circle((-0.5, 0.5), 0.25, fill=False)
        circle3 = plt.Circle((0.5, -0.5), 0.25, fill=False)
        circle4 = plt.Circle((0.5, 0.5), 0.25, fill=False)
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)
        plt.gcf().gca().add_artist(circle4)

        x_traj = []
        y_traj = []
        for point in self.turn:
            x_traj.append(point[1][0])
            y_traj.append(point[1][1])

        plt.plot(x_traj, y_traj, marker='o')
        plt.title(title)

        # pylab.savefig('condRW')

    def finalPositions(self):
        tempList = []
        tempList.append(tuple(self.turn[-1][1:]))
        # print(tempList)
        # print(self.turn)
        return tempList

    def weight(self, time):
        """Compute Particle Weight at time t."""
        i = 0
        maxI = len(self.turn)-1
        maxTime = self.turn[-1][0]
        if maxTime < time:
            # print("Warning: Time too big in hRW.weight()")
            # print(time, maxTime)
            # print(self.turn)
            return 0.
        else:
            while (i < maxI - 1 and self.turn[i + 1][0] < time):
                i += 1
            oldPos = self.turn[i][1]
            nextPos = self.turn[i+1][1]
            y = (time-self.turn[i][0])/(self.turn[i+1][0]-self.turn[i][0])
            eventPos = [oldPos[0] + y*(nextPos[0]-oldPos[0]),
                        oldPos[1] + y*(nextPos[1]-oldPos[1])]
            wt = self.turn[i][3] * np.exp(self.h.intJhBeta(oldPos, eventPos))
        return wt
