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
from tqdm import tqdm

from scipy import stats
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

from NTE2D import PathCircles
from simulate_tools import SimulateFunction

from hTransf import hRW
from ParticleFilter import PF

import time

import os
import sys
from pathlib import Path

from Logger import Logger

# Choose which plots to produce.
AnalysehTransfPF = False
PlotAnim = False
CompareEigenEstimates = True
PlotEigenfnEstimates = False

# Log the output.
useLogger = True
useGitDiff = True

if useLogger:
    id_str = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.getcwd()+'/output/'+id_str
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.rcParams['figure.figsize'] = [8.0, 3.2]
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 'small'
    plt.rcParams['figure.titlesize'] = 'medium'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    # NB: Latex needs to be in path, or set usetex = False.
    # May need to run:
    #   import os ;;
    #   os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
    # Or similar.


if useLogger and useGitDiff:
    stream = os.popen('git log -n 1')
    open(output_dir+"/git_version_info.txt", "a").write(stream.read())
    stream = os.popen('git diff')
    open(output_dir+"/git_version_info.txt", "a").write(stream.read())

startTime = time.time()

if useLogger:
    sys.stdout = Logger()

    print("\n*****************")
    print(id_str)
    print("Output Directory: ", output_dir)
    print("Main Script: ", __file__)
    print("*****************")

if AnalysehTransfPF:
    Srate_dom = 2.5  # Scatter rate for underlying domain
    Srate_circ = 5.0  # Scatter rate for circles
    Brate_dom = 0.1  # Branching rate for underlying domain
    Brate_circ = 1.25  # Branching rate for circles

    v_start = 1

    UsehTransf = True  # Simulate using h-Transformed particles.

    def initPos():
        """Set the initial position of the Simulation."""
        # return ([-0.9,-0.9],np.pi/4.)
        return (list(np.random.uniform(-1., 1., 2)),
                np.random.uniform(0., 2 * np.pi))

    def motion(pos, theta, timeStep):
        """Set the motion of the particles."""
        if UsehTransf:
            return hRW(0.0, timeStep, pos, theta, Srate_dom, Srate_circ,
                       Brate_dom, Brate_circ, v_start, c=0.1, C=5.0)
        else:
            return PathCircles(0.0, timeStep, pos, theta, Srate_dom,
                               Srate_circ, Brate_dom, Brate_circ, v_start)

    NoPart = 250

    pf = PF(initPos, motion, nPart=NoPart)
    # print(pf.weights())

    print("\n~~~~ Simulation of Particle Filter + h-Transform ~~~~\n")

    print("Number of Particles: " + str(NoPart) + "\n")

    print("\n~~~~ First Iteration ~~~~\n")

    pf.mutate(3.)
    # print(pf.weights())
    with plt.rc_context({'figure.figsize': [4., 4.]}):
        pf.plot()
        plt.title("Particle Positions, t=3.0")

        if useLogger:
            plt.savefig(output_dir+'/PF3.pdf', format='pdf',
                        bbox_inches="tight")

    # print("Weights: ",pf.weights())
    # print("Effective Sample Size: ",pf.effectiveSampleSize())

    print("\n~~~~ Second Iteration ~~~~\n")

    pf.resample()
    pf.mutate(3.)
    # print(pf.weights())

    with plt.rc_context({'figure.figsize': [4., 4.]}):
        pf.plot()
        plt.title("Particle Positions, t=6.0")

        if useLogger:
            plt.savefig(output_dir+'/PF6.pdf', format='pdf',
                        bbox_inches="tight")

    # print("Weights: ",pf.weights())
    # print("Effective Sample Size: ",pf.effectiveSampleSize())

    print("\n~~~~ Final Iteration (long) ~~~~\n")

    pf.resample()
    pf.step(5, 3.0)
    # print(pf.weights())

    with plt.rc_context({'figure.figsize': [4., 4.]}):
        pf.plot()
        plt.title("Particle Positions, t=21.0")

        if useLogger:
            plt.savefig(output_dir+'/PF21.pdf', format='pdf',
                        bbox_inches="tight")

    plt.figure()
    plt.plot(pf.timeVec(), pf.weightOverTime())
    plt.title("Weight of particle system over time")

    if useLogger:
        plt.savefig(output_dir+'/PFWeight.pdf', format='pdf')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        pf.timeVec(), np.log(pf.weightOverTime()))

    plt.figure()
    plt.plot(pf.timeVec(), intercept + pf.timeVec() * slope, 'r--',
             pf.timeVec(), np.log(pf.weightOverTime()), 'b')
    plt.title("log-plot of particle system weight, with linear fit, slope = " +
              str(slope))

    if useLogger:
        plt.savefig(output_dir+'/PFLogWeight.pdf', format='pdf')

    print("Estimated Eigenvalue: ", np.exp(slope))

    plt.figure()
    plt.plot(pf.timeVec()[1:], pf.ESSOverTime())
    plt.title("ESS of particle system over time")

    if useLogger:
        plt.savefig(output_dir+'/PFESS.pdf', format='pdf')

    # plt.figure()
    with plt.rc_context({'figure.figsize': [4., 4.]}):
        if useLogger:
            pf.heatMap(filename=output_dir+'/PFHeatMap.pdf')
        else:
            pf.heatMap()

    if PlotAnim:
        print("\n~~~~ Producing Animation ~~~~\n")

        if useLogger:
            ani = pf.heatMap2(useWeights='Normalised', resampleEvery=5,
                              tStep=0.25, nStepsToGo=20,
                              filename=output_dir+"/PFHeatmap.gif")
        else:
            ani = pf.heatMap2(useWeights='Normalised', resampleEvery=5,
                              tStep=0.25, nStepsToGo=20)

        plt.show(ani)

if CompareEigenEstimates:
    T_0 = 0             # Starting time
    T = 40              # Final time
    X_0 = [-0.9, -0.9]  # Initial position
    Theta_0 = np.pi/3   # Initial direction
    Srate_dom = 2.5     # Scatter rate for underlying domain
    Srate_circ = 5.0    # Scatter rate for circles
    Brate_dom = 0.1     # Branching rate for underlying domain
    Brate_circ = 1.25   # Branching rate for circles

    t_start = 0
    v_start = 1
    sample_times = np.linspace(0.05*T, 0.99*T, 20)
    n_sim_br = 400
    n_sim_nrw = 50000
    n_sim_h = 200
    n_rep = 5

    mot2_vals = [0.001, 0.5, 0.5]

    print("\n~~~~ Compare NBP, NRW, hRW performance ~~~~\n")

    print("*****************")
    print("*****************")
    print("Scatter Rate in Domain = ", Srate_dom,
          "\nBranch Rate in Domain = ", Brate_dom,
          "\nScatter Rate in Circle = ", Srate_circ,
          "\nBranch Rate in Circle = ", Brate_circ)

    print("*****************")
    print("Number of Sims (NBP) = ", n_sim_br,
          "\nNumber of Sims (NRW) = ", n_sim_nrw,
          "\nNumber of Sims (h-RW) = ", n_sim_h,
          )
    print("*****************")
    print("h-Transform conditioning values (c, C, slope): ", mot2_vals)
    print("*****************")


    fun = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun2 = lambda : fun(PathCircles(T_0, T, X_0, Theta_0, Srate_dom,
                                    Srate_circ, Brate_dom, Brate_circ,
                                    v_start))

    fun3 = lambda arg: np.array([arg.weight(arg2) for arg2 in sample_times])
    fun4 = lambda : fun3(hRW(T_0, T, X_0, Theta_0, Srate_dom, Srate_circ,
                             Brate_dom, Brate_circ, v_start,
                             c=mot2_vals[0], C=mot2_vals[1],
                             slope=mot2_vals[2]))

    fun5 = lambda arg: np.array([arg.Integral(Brate_dom, Brate_circ, arg2)
                                 for arg2 in sample_times])
    fun6 = lambda : fun5(PathCircles(T_0, T, X_0, Theta_0, Srate_dom
                                     + Brate_dom, Srate_circ + Brate_circ, 0,
                                     0, v_start))

    # =========================================================================
    # Simple branching
    # =========================================================================
    print("\nSimple branching")
    print("===================")

    Ei = np.zeros([n_rep, len(sample_times)])

    for i in tqdm(range(n_rep)):
        sim = SimulateFunction(fun2, n_sim_br)
        with np.errstate(divide='ignore'):
            Ei[i] = np.divide(np.log(np.mean(sim.z, 0)), sample_times)

    Av = np.mean(Ei, 0)
    Var = np.var(Ei, 0)

    print("\nEstimate of eigenvalue at sample times:", Av)
    print("\nVariance at sample times:", Var)

    # =========================================================================
    # Vanilla Many-to-one
    # =========================================================================
    print("\nMany-to-one")
    print("===================")

    Ei2 = np.zeros([n_rep, len(sample_times)])

    for i in tqdm(range(n_rep)):
        sim2 = SimulateFunction(fun6, n_sim_nrw)
        with np.errstate(divide='ignore'):
            Ei2[i] = np.divide(np.log(np.mean(sim2.z, 0)), sample_times)

    Av2 = np.mean(Ei2, 0)
    Var2 = np.var(Ei2, 0)

    print("\nEstimate of eigenvalue at sample times:", Av2)
    print("\nVariance at sample times:", Var2)

    # =========================================================================
    # Many-to-one with h-transform
    # =========================================================================
    print("\nh-transformed RW")
    print("===================")
    Ei3 = np.zeros([n_rep, len(sample_times)])

    for i in tqdm(range(n_rep)):
        sim = SimulateFunction(fun4, n_sim_h)
        with np.errstate(divide='ignore'):
            Ei3[i] = np.divide(np.log(np.mean(sim.z, 0)), sample_times)

    Av3 = np.mean(Ei3, 0)
    Var3 = np.var(Ei3, 0)

    print("\nEstimate of eigenvalue at sample times:", Av3)
    print("\nVariance at sample times:", Var3)
    with plt.rc_context({'figure.figsize': [4., 3.2]}):
        plt.figure()
        br, = plt.plot(sample_times, Av, '-', markersize=4, label=r'NBP')
        con, = plt.plot(sample_times, Av3, label=r'$h$-RW')
        sm, = plt.plot(sample_times, Av2, '-', markersize=4,  label=r'NRW')

        plt.legend(handles=[br, sm, con])

        # plt.title(r"Estimates of eigenvalue over time")
        plt.xlabel(r"Time")

        if useLogger:
            plt.savefig(output_dir+'/2DCompare.pdf', format='pdf')

if PlotEigenfnEstimates:
    T_0 = 0            # Starting time
    T = 50             # Final time
    X_0 = [0, 0]       # Initial position
    Theta_0 = np.pi/4  # Initial direction
    V_0 = 1            # Initial Velocity
    Srate_dom = 2.5    # Scatter rate for underlying domain
    Srate_circ = 5.0   # Scatter rate for circles
    Brate_dom = 0.1    # Branching rate for underlying domain
    Brate_circ = 1.25  # Branching rate for circles

    sample_times = np.linspace(0.2*T, 0.99*T, 10)

    nreps = 500

    print("~~~~ Produce a plot of the Eigenfunction (Experimental) ~~~~\n")

    print("*****************")
    print("*****************")
    print("Scatter Rate in Domain = ", Srate_dom,
          "\nBranch Rate in Domain = ", Brate_dom,
          "\nScatter Rate in Circle = ", Srate_circ,
          "\nBranch Rate in Circle = ", Brate_circ)

    ###########################################################################
    # Estimate the eigenvalue
    ###########################################################################

    print("~~~~ Estimating Eigenvalue ~~~~\n")

    fun = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun2 = lambda : fun(PathCircles(T_0, T, X_0, Theta_0, Srate_dom,
                                    Srate_circ, Brate_dom, Brate_circ, V_0))

    Ei = np.divide(np.log(np.mean(SimulateFunction(fun2, nreps,
                                                   ProgBar=True).z, 0)),
                   sample_times)

    plt.figure()
    plt.plot(sample_times, Ei)

    print("Eigenvalue Estimate: ", Ei)

    if useLogger:
        plt.savefig(output_dir+'/2DEigenval.pdf', format='pdf')

    ###########################################################################
    # Estimate the eigenfunction at a given angle
    ###########################################################################
    print("~~~~ Estimating Eigenfunction ~~~~\n")

    step = 22
    n_sim = 200

    print("*****************")
    print("Number of Sims per site = ", n_sim)
    print("Number of sites = ", (step-2)**2)
    print("*****************")

    X = np.linspace(-1, 1, step)
    X = X[1:-1]
    Y = np.linspace(-1, 1, step)
    Y = Y[1:-1]

    phi = np.array([np.zeros(len(X)) for j in range(len(Y))])

    fun3 = lambda arg: PathCircles(T_0, T, arg, Theta_0, Srate_dom, Srate_circ,
                                   Brate_dom, Brate_circ, V_0).count(
                                       sample_times) * np.exp(-Ei)

    for i in tqdm(range(len(X))):
        for j in range(len(X)):
            z = np.zeros(n_sim)
            for k in range(n_sim):
                z[k] = np.mean(fun3([X[i], Y[j]]), 0)
            phi[i][j] = np.mean(z, 0)

    print("\nPhi", phi)

    ###########################################################################
    # Plotting
    ###########################################################################

    with plt.rc_context({'figure.figsize': [4., 4.]}):
        phi *= 1 / np.sum(phi)
        XX, YY = np.meshgrid(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(XX, YY, phi, s=10)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_wireframe(XX, YY, phi)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot_surface(XX, YY, phi, rstride=1, cstride=1, cmap='viridis',
                         edgecolor='none')

    ###########################################################################
    # colormap
    ###########################################################################

    with plt.rc_context({'figure.figsize': [4., 4.]}):
        phi2 = phi[:-1, :-1]
        XXX, YYY = np.meshgrid(X[:-1], Y[:-1])

        fig4, ax4 = plt.subplots()
        p = ax4.pcolor(XXX, YYY, phi2, cmap=plt.cm.jet, vmin=abs(phi2).min(),
                       vmax=abs(phi2).max())
        cb4 = fig4.colorbar(p)

    ###########################################################################
    # Smoothing
    ###########################################################################

    with plt.rc_context({'figure.figsize': [4., 4.]}):
        Z = gaussian_filter(phi, sigma=20)
        extent = [X[0], X[-1], Y[0], Y[-1]]
        img = Z.T

        fig5 = plt.figure()
        ax5 = fig5.add_axes([0.1, 0.1, 1.5, 1.5])
        ax5.imshow(img, extent=extent, origin='lower', cmap=plt.cm.jet)
        cb4 = fig5.colorbar(p)

        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111, projection='3d')
        ax6.plot_surface(XX, YY, Z.T, rstride=1, cstride=1, cmap='viridis',
                         edgecolor='none')

        if useLogger:
            fig.savefig(output_dir+'/2DEFn1.pdf', format='pdf')
            fig2.savefig(output_dir+'/2DEFn2.pdf', format='pdf')
            fig3.savefig(output_dir+'/2DEFn3.pdf', format='pdf')
            fig4.savefig(output_dir+'/2DEFn4.pdf', format='pdf')
            fig5.savefig(output_dir+'/2DEFn5.pdf', format='pdf',
                         bbox_inches="tight")
            fig6.savefig(output_dir+'/2DEFn6.pdf', format='pdf')

print("\nTotal Time Taken: ", round(time.time()-startTime), " seconds")

if useLogger:
    sys.stdout.close()
