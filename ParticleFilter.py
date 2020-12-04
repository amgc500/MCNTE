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

# Implement a `particle filter'-like class

import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from random import choices
# import time
from tqdm import tqdm
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
import matplotlib.animation as animation


class PF:
    """Run a particle Filter on a given motion."""

    def __init__(self, initPos, motion,
                 nPart=100, recordSelectionSites=False):
        self.nPart = nPart
        self.initPos = initPos
        self.motion = motion
        # class of form motion(pos,theta,timeStep)
        # Class should have member functions:
        #    finalPositions, list of particles in form
        #    [position, theta, weight]
        # self.domain = domain

        self.tCurrent = 0.
        self.nStep = 0

        self.totalWeight = [1.]
        self.time = [self.tCurrent]
        self.particles = []
        self.ess = []
        self.recordSelectionSites = recordSelectionSites
        if self.recordSelectionSites:
            self.birthSites = []
            self.deathSites = []

        for i in range(nPart):
            self.particles.append((*self.initPos(),
                                   self.totalWeight[-1]/self.nPart))

    def mutate(self, tStep=1.0):
        """
        Evolve the particle system for a time tStep.

        Parameters
        ----------
        tStep : TYPE, optional
            How long to simulate process for. The default is 1.0.
        """
        if self.particles:
            # Check that there are particles still alive. Otherwise do nothing.
            tempParticles = []
            for p in self.particles:
                p2 = self.motion(p[0], p[1], tStep).finalPositions()
                for p3 in p2:
                    tempParticles.append((p3[0], p3[1], p3[2] * p[2]))
                    # tempParticles.append((p3[0], p3[1],
                    #                       p3[2] * self.totalWeight[-1] /
                    #                       self.nPart))
            self.particles = tempParticles

            self.totalWeight.append(np.sum([p[2] for p in self.particles]))
            self.ess.append(self.effectiveSampleSize())
            self.tCurrent += tStep
            self.time.append(self.tCurrent)
            self.nStep += 1

    def plot(self, title=""):
        """
        Plot the positions of the particles in the current state.

        Parameters
        ----------
        title : TYPE, optional
            Plot title. The default is "".
        """
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

        for p in self.particles:
            plt.plot(p[0][0], p[0][1], marker='o')
        plt.title(title)

    def plotBirthDeath(self, title="", thetaRange=(-np.pi, np.pi)):
        """
        Plot the positions of all Birth/Death sites so far.

        Parameters
        ----------
        title : TYPE, optional
            Plot title. The default is "".

        thetaRange : (real, real), optional
            Will only plot points which have angle between the angles.
        """
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

        eps = 0.02

        bSVec = np.zeros((2, len(self.birthSites)))
        bSCount = 0

        for p in self.birthSites:
            pos = p[0]
            theta = p[1]

            if (theta >= thetaRange[0]) and (theta <= thetaRange[1]):
                jitx = np.random.uniform(-eps, eps)
                jity = np.random.uniform(-eps, eps)
                bSVec[:, bSCount] = (pos[0] + jitx, pos[1] + jity)
                bSCount += 1

        plt.plot(bSVec[0, :bSCount], bSVec[1, :bSCount], marker='o',
                 color='tab:blue', alpha=0.3, label="Birth", linestyle="None")

        dSVec = np.zeros((2, len(self.deathSites)))
        dSCount = 0

        for p in self.deathSites:
            pos = p[0]
            theta = p[1]

            if (theta >= thetaRange[0]) and (theta <= thetaRange[1]):
                jitx = np.random.uniform(-eps, eps)
                jity = np.random.uniform(-eps, eps)
                dSVec[:, dSCount] = (pos[0] + jitx, pos[1] + jity)
                dSCount += 1

        plt.plot(dSVec[0, :dSCount], dSVec[1, :dSCount], marker='o',
                 color='tab:red', alpha=0.3, label="Death", linestyle="None")

        plt.title(title)
        plt.legend()

    def weights(self):
        """Return the particle weights."""
        # if isinstance(self.motion((0.0, 0.0), 0., 0.), hRW):
        #     h = self.motion((0.0, 0.0), 0., 0.).h.val
        #     return np.array([p[2] * h(p[0], p[1]) for p in self.particles])
        # else:
        #     return np.array([p[2] for p in self.particles])
        return np.array([p[2] for p in self.particles])

    def effectiveSampleSize(self):
        """Return the effective sample size of current particles."""
        w = self.weights()
        if not w.size == 0:
            return (np.sum(w)**2.) / np.sum(np.power(w, 2.))
        else:
            return 0.

    def resample(self):
        """
        Resample the particle population.

        Resample the particle population to get nPart equally weighted
        particles. Note that the total particle weight will remain
        unchanged.

        Returns
        -------
        list (self.particles)
            List of particle positions, weights.

        """
        # Check that there are particles alive:
        if self.particles:
            # If self.recordSelectionSites = True, we need to also compute the
            #   locations of particles which are not
            if self.recordSelectionSites:
                tempParticles = choices(np.arange(len(self.particles)),
                                        weights=self.weights(), k=self.nPart)
                # tempParticles = choices(zip(self.particles,
                #                             range(len(self.particles))),
                #                         weights=self.weights(), k=self.nPart)
                tempCount = np.zeros(len(self.particles), dtype='int')

                tempParticles2 = []
                for i in tempParticles:
                    tempParticles2.append((*self.particles[i][:2],
                                           self.totalWeight[-1]/self.nPart))
                    tempCount[i] += 1

                for i in range(len(self.particles)):
                    if tempCount[i] == 0:
                        self.deathSites.append(self.particles[i][:2])
                    elif tempCount[i] > 1.5:
                        for j in range(tempCount[i]-1):
                            self.birthSites.append(self.particles[i][:2])
                self.particles = tempParticles2

            else:
                tempParticles = choices(self.particles, weights=self.weights(),
                                        k=self.nPart)
                self.particles = []
                for i in range(self.nPart):
                    self.particles.append((*tempParticles[i][:2],
                                           self.totalWeight[-1]/self.nPart))

        return self.particles

    def step(self, nStepsToGo=1, tStep=1.0):
        """
        Perform a single/multple mutate/resample steps of the particle system.

        Parameters
        ----------
        nStepsToGo : TYPE, optional
            DESCRIPTION. The default is 1.
        tStep : TYPE, optional
            DESCRIPTION. The default is 1.0.

        Returns
        -------
        list (self.particles)
            List of particle positions, weights.

        """
        for i in tqdm(range(nStepsToGo)):
            self.mutate(tStep)
            self.resample()
        return self.particles

    def weightOverTime(self):
        """
        Return the total weight of the particle system.

        Returns
        -------
        numpy array
            Total weight of particle system at each time.

        """
        return np.array(self.totalWeight)

    def ESSOverTime(self):
        """
        Return the effective sample size (ESS) of the particle system.

        Returns
        -------
        numpy array
            Total ESS particle system at each time.

        """
        return np.array(self.ess)

    def timeVec(self):
        return np.array(self.time)

    def heatMap(self, s=100., useWeights=True, filename=""):
        x = []
        y = []
        w = []
        for p in self.particles:
            x.append(p[0][0])
            y.append(p[0][1])
            w.append(p[2])

        if useWeights:
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000,
                                                     weights=w)
        else:
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=1000)

        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = heatmap.T

        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 2, 2])
        axes.set_aspect('equal')
        plt.xlim(-1., 1.)
        plt.ylim(-1., 1.)
        axes.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
        axes.set_title(r"Smoothing with  $\sigma$ = %d" % s)

        if filename:
            plt.savefig(filename, format='pdf', bbox_inches="tight")

        plt.show()

    def heatMap2(self, s=100., resampleEvery=1, useWeights='Normalised',
                 nStepsToGo=20, tStep=1.0, filename="Anim.gif"):
        x = []
        y = []
        w = []
        for p in self.particles:
            x.append(p[0][0])
            y.append(p[0][1])
            w.append(p[2])

        if useWeights == 'Normalised' or useWeights == 'Raw':
            heatmap, xedges, yedges = np.histogram2d(x, y, range=[[-1., 1.],
                                                                  [-1., 1.]],
                                                     bins=1000, weights=w)
        else:
            heatmap, xedges, yedges = np.histogram2d(x, y, range=[[-1., 1.],
                                                                  [-1., 1.]],
                                                     bins=1000)

        w0 = np.sum(w)

        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        img = heatmap.T

        fig = plt.figure(figsize=[10., 10.])
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.set_title(r"Smoothing with  $\sigma$ = %d" % s)
        circle1 = plt.Circle((-0.5, -0.5), 0.25, fill=False)
        circle2 = plt.Circle((-0.5, 0.5), 0.25, fill=False)
        circle3 = plt.Circle((0.5, -0.5), 0.25, fill=False)
        circle4 = plt.Circle((0.5, 0.5), 0.25, fill=False)
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)
        plt.gcf().gca().add_artist(circle4)
        im = ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet,
                       animated=True)

        i = 0

        def updateFig(*args):
            nonlocal i
            printChr = ['|','\\','-','/','*\n']
            print(printChr[i], end='')
            i += 1

            self.mutate(tStep)
            if i == resampleEvery:
                self.resample()
                i = 0

            x = []
            y = []
            w = []
            for p in self.particles:
                x.append(p[0][0])
                y.append(p[0][1])
                w.append(p[2])

            if useWeights == 'Normalised':
                wFactor = w0/np.sum(w)
                heatmap, xedges, yedges = np.histogram2d(x, y,
                                                         range=[[-1., 1.],
                                                                [-1., 1.]],
                                                         bins=1000,
                                                         weights=np.array(w)
                                                         * wFactor)
            elif useWeights == 'Raw':
                heatmap, xedges, yedges = np.histogram2d(x, y,
                                                         range=[[-1., 1.]
                                                                [-1., 1.]],
                                                         bins=1000,
                                                         weights=w)
            else:
                heatmap, xedges, yedges = np.histogram2d(x, y,
                                                         range=[[-1., 1.],
                                                                [-1., 1.]],
                                                         bins=1000)

            heatmap = gaussian_filter(heatmap, sigma=s)

            circle1 = plt.Circle((-0.5, -0.5), 0.25, fill=False)
            circle2 = plt.Circle((-0.5, 0.5), 0.25, fill=False)
            circle3 = plt.Circle((0.5, -0.5), 0.25, fill=False)
            circle4 = plt.Circle((0.5, 0.5), 0.25, fill=False)
            plt.gcf().gca().add_artist(circle1)
            plt.gcf().gca().add_artist(circle2)
            plt.gcf().gca().add_artist(circle3)
            plt.gcf().gca().add_artist(circle4)

            img = heatmap.T
            im.set_array(img)
            return im,

        ani = animation.FuncAnimation(fig, updateFig, interval=100, blit=True,
                                      frames=nStepsToGo)

        ani.save(filename)
        return ani

# def animateHeatMap(self, nStepsToGo = 1, tStep = 1.0, s=100.,
#      useWeights=True):
