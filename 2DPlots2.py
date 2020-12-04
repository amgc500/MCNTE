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
import time
import os
from scipy import stats
from NTE2D import PathCircles
from hTransf import hRW, hFunction
from ParticleFilter import PF
import sys
from pathlib import Path
from math import floor

from Logger import Logger

useLogger = True
useGitDiff = True
plotTitles = False

T = 200             # Final time
Srate_dom = 2.5     # Scatter rate for underlying domain
Srate_circ = 5.0    # Scatter rate for circles
Brate_dom = 0.1     # Branching rate for underlying domain
Brate_circ = 1.25   # Branching rate for circles

v_start = 1

numPart = 100
numSteps = 1600
tStep = T/numSteps

if useLogger:
    id_str = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.getcwd()+'/output/'+id_str
    Path(output_dir).mkdir(parents=True, exist_ok=True)


if useLogger and useGitDiff:
    stream = os.popen('git log -n 1')
    open(output_dir+"/git_version_info.txt", "a").write(stream.read())
    stream = os.popen('git diff')
    open(output_dir+"/git_version_info.txt", "a").write(stream.read())

startTime = time.time()


def initPos():
    """Return a (possibly random) starting point."""
    # return ([-0.9,-0.9],np.pi/4.)
    return (list(np.random.uniform(-1., 1., 2)),
            np.random.uniform(-np.pi, np.pi))


def motion(pos, theta, timeStep):
    """Define a BRW motion for the mutation step."""
    return PathCircles(0.0, timeStep, pos, theta, Srate_dom, Srate_circ,
                       Brate_dom, Brate_circ, v_start)


mot2_vals = [0.00001, 0.1, 0.02]


def motion2(pos, theta, timeStep):
    """Define a h-transformed motion for the mutation step."""
    return hRW(0.0, timeStep, pos, theta, Srate_dom, Srate_circ, Brate_dom,
               Brate_circ, v_start, c=mot2_vals[0], C=mot2_vals[1],
               slope=mot2_vals[2])


def motion3(pos, theta, timeStep):
    """Define a weakly h-transformed motion for the mutation step."""
    return hRW(0.0, timeStep, pos, theta, Srate_dom, Srate_circ, Brate_dom,
               Brate_circ, v_start, c=25.0, C=25.0, slope=1.)


pfs = (PF(initPos, motion, nPart=numPart, recordSelectionSites=True),
       PF(initPos, motion2, nPart=numPart, recordSelectionSites=True),
       PF(initPos, motion3, nPart=numPart, recordSelectionSites=True))

pfNames = (r"Branching Monte Carlo", r"$h$-RW", "moderate $h$-NRW")

# print(pf.weights())

for pf in pfs:
    pf.step(numSteps, tStep)

if useLogger:
    sys.stdout = Logger()

    print("\n*****************")
    print(id_str)
    print("Output Directory: ", output_dir)
    print("Main Script: ", __file__)
    print("*****************")


print("\nNumber of Particles = ", numPart,
      "\nNumber of Steps = ", numSteps,
      "\nTime Step = ", tStep)


print("\nScatter Rate in Domain = ", Srate_dom,
      "\nBranch Rate in Domain = ", Brate_dom,
      "\nScatter Rate in Circle = ", Srate_circ,
      "\nBranch Rate in Circle = ", Brate_circ)

print("h-Transform conditioning values (c, C, slope): ", mot2_vals)

burnIn = 0.3
N0 = round(burnIn*numSteps)
tailEstimate = 0.5
N1 = round(tailEstimate*numSteps)
print("Burn In Period: t <= ", burnIn * numSteps * tStep)

i = 0
with plt.rc_context({'figure.figsize': [4., 3.2]}):
    plt.figure()
    for pf in pfs:
        plt.plot(pf.timeVec()[N1:], np.log(pf.weightOverTime()[N1:] /
                                           pf.weightOverTime()[N0]),
                 label=pfNames[i])
        slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pf.timeVec()[N0:], np.log(pf.weightOverTime()[N0:]))
        plt.xlabel(r"Time")
        print("Estimated Eigenvalue (lambda): ", slope, " (", pfNames[i], ")")
        if plotTitles:
            plt.title("log-Weight vs Time (" + id_str + ")")
        i += 1

    plt.legend()

    if useLogger:
        plt.savefig(output_dir+'/weights_1.pdf', format='pdf')

i = 0
plt.figure()
for pf in pfs:
    plt.plot(pf.timeVec()[N1:], np.log(pf.weightOverTime()[N1:] /
                                       pf.weightOverTime()[N0]),
             label=pfNames[i])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
                pf.timeVec()[N0:], np.log(pf.weightOverTime()[N0:]))
    plt.xlabel(r"Time")
    print("Estimated Eigenvalue (lambda): ", slope, " (", pfNames[i], ")")
    if plotTitles:
        plt.title("log-Weight vs Time (" + id_str + ")")
    i += 1

plt.legend()

if useLogger:
    plt.savefig(output_dir+'/weights_1_wide.pdf', format='pdf')

i = 0
with plt.rc_context({'figure.figsize': [4., 3.2]}):
    plt.figure()
    for pf in pfs:
        plt.plot(pf.timeVec()[N1:], np.log(pf.weightOverTime()[N1:] /
                                           pf.weightOverTime()[N0]) /
                 (pf.timeVec()[N1:] - pf.timeVec()[N0]), label=pfNames[i])
        if plotTitles:
            plt.title("Method 1[GM]: Simple Estimate vs Time (" + id_str + ")")
        i += 1

    plt.legend()
    plt.xlabel(r"Time")

    if useLogger:
        plt.savefig(output_dir+'/weights_1_5.pdf', format='pdf')

i = 0
plt.figure()
for pf in pfs:
    plt.plot(pf.timeVec()[N1:], np.log(pf.weightOverTime()[N1:] /
                                       pf.weightOverTime()[N0]) /
             (pf.timeVec()[N1:] - pf.timeVec()[N0]), label=pfNames[i])
    if plotTitles:
        plt.title("Method 1[GM]: Simple Estimate vs Time (" + id_str + ")")
    i += 1

plt.legend()
plt.xlabel(r"Time")

if useLogger:
    plt.savefig(output_dir+'/weights_1_5_wide.pdf', format='pdf')


# slope, intercept, r_value, p_value, std_err = stats.linregress(
#     pf.timeVec(), np.log(pf.weightOverTime()))

# plt.figure()
# plt.plot(pf.timeVec(), intercept + pf.timeVec() * slope, 'r--',
#          pf.timeVec(), np.log(pf.weightOverTime()), 'b')
# print("Estimated Eigenvalue: ", np.exp(slope))

# plt.figure()
# plt.plot(pf.timeVec()[1:], pf.ESSOverTime())

# # plt.figure()
# pf.heatMap()

# ani = pf.heatMap2(useWeights='Normalised', resampleEvery=5, tStep=0.25,
#                   nStepsToGo=20)

# plt.show(ani)

i = 0
with plt.rc_context({'figure.figsize': [4., 3.2]}):
    plt.figure()
    for pf in pfs:
        temp = pf.weightOverTime()[(N0+1):]/pf.weightOverTime()[N0:-1]
        temp = np.cumsum(temp)/np.arange(1, numSteps + 1 - N0)
        plt.plot(pf.timeVec()[N1:-1], np.log(temp[(N1 - N0):]) / tStep,
                 label=pfNames[i])
        if plotTitles:
            plt.title("Method 2 [AM]: Simple Estimate vs Time (" + id_str + ")")
        print("Estimated Eigenvalue (lambda) (Method 2 [AM]): ",
              np.log(temp[-1])/tStep,
              " (", pfNames[i], ")")
        i += 1

    plt.legend()
    plt.xlabel(r"Time")

    if useLogger:
        plt.savefig(output_dir+'/weights_2.pdf', format='pdf')


i = 0
plt.figure()
for pf in pfs:
    temp = pf.weightOverTime()[(N0+1):]/pf.weightOverTime()[N0:-1]
    temp = np.cumsum(temp)/np.arange(1, numSteps + 1 - N0)
    plt.plot(pf.timeVec()[N1:-1], np.log(temp[(N1 - N0):]) / tStep,
             label=pfNames[i])
    if plotTitles:
        plt.title("Method 2 [AM]: Simple Estimate vs Time (" + id_str + ")")
    print("Estimated Eigenvalue (lambda) (Method 2 [AM]): ",
          np.log(temp[-1])/tStep,
          " (", pfNames[i], ")")
    i += 1

plt.legend()
plt.xlabel(r"Time")

if useLogger:
    plt.savefig(output_dir+'/weights_2_wide.pdf', format='pdf')

i = 0
for pf in pfs:
    plt.figure()
    if plotTitles:
        pf.plot(title="Final Positions of "+pfNames[i])
    i += 1

plt.figure()

N0 = round(0.5*numSteps)

print("Time interval to compute estimates: [", N0*tStep, ", ", numSteps*tStep,
      "]")

i = 0

dat = np.zeros((numSteps-N0+1, len(pfs)))

for pf in pfs:
    print("\n", pfNames[i], ":")
    print("\n*****************")
    dat[:, i] = pf.weightOverTime()[N0:]/pf.weightOverTime()[N0-1:-1]
    print("AM 95% Confidence interval: (",
          np.log(np.mean(dat[:, i]) - 1.96 * (np.std(dat[:, i]) /
                                              np.sqrt(numSteps-N0)))/tStep,
          ", ",
          (np.log(np.mean(dat[:, i]) + 1.96 * (np.std(dat[:, i]) /
                                               np.sqrt(numSteps-N0)))
           / tStep), " ) Median: ", np.log(np.mean(dat[:, i])) / tStep)
    print("GM 95% Confidence interval: (",
          np.mean(np.log(dat[:, i]))/tStep - 1.96 * np.std(np.log(dat[:, i]))
          / np.sqrt(numSteps-N0), ", ",
          np.mean(np.log(dat[:, i]))/tStep + 1.96 * np.std(np.log(dat[:, i]))
          / np.sqrt(numSteps-N0), " ) Median: ", (np.mean(np.log(dat[:, i]))
                                                  / tStep))
    i += 1

plt.hist(np.log(dat), bins=20)
if plotTitles:
    plt.title("Histogram of log-weight (proportionate) change per step.")
plt.legend(pfNames)
if useLogger:
    plt.savefig(output_dir+'/log_weights_hist.pdf', format='pdf')
plt.figure()
plt.hist(dat, bins=20)
if plotTitles:
    plt.title("Histogram of weight (proportionate) change per step.")
plt.legend(pfNames)
if useLogger:
    plt.savefig(output_dir+'/weights_hist.pdf', format='pdf')

plt.figure()
f, axes = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(30, 10))


i = 0
cmap = plt.get_cmap('Set1')
for pf in pfs:
    axes[i].plot(dat[:, i], color=cmap(i / len(pfs)))
    if plotTitles:
        axes[i].set_title("Weight Change ("+pfNames[i]+")")
    i += 1

if useLogger:
    plt.savefig(output_dir+'/weights_changes.pdf', format='pdf')

i = 0
for pf in pfs:
    pf.plotBirthDeath("Particle Birth/Death ("+pfNames[i]+")")
    plt.autoscale()
    if useLogger:
        plt.savefig(output_dir+'/birth_death_'+pfNames[i]+'.pdf', format='pdf',
                    bbox_inches="tight")
    i += 1

i = 0
J = 4

for pf in pfs:
    for j in range(J):
        pf.plotBirthDeath("Particle Birth/Death ("+pfNames[i]+")",
                          ((j / J) * 2 * np.pi - np.pi,
                           ((j + 1) / J) * 2 * np.pi - np.pi))
        if useLogger:
            plt.savefig(output_dir+'/birth_death_angular'+pfNames[i]+str(j)
                        + '.pdf', format='pdf', bbox_inches="tight")
    i += 1

i = 1
for pf in pfs[i:]:
    hBirthSum = 0.
    hDeathSum = 0.
    h = pf.motion((0., 0.), 0., 0.).h
    for p in pf.birthSites:
        hBirthSum += h.val(*p)
    for p in pf.deathSites:
        hDeathSum += h.val(*p)
    print(pfNames[i]+" average birth sites h-value: ",
          hBirthSum/len(pf.birthSites))
    print(pfNames[i]+" average death sites h-value: ",
          hDeathSum/len(pf.deathSites))
    print("Ratio: ", hBirthSum/len(pf.birthSites) /
          (hDeathSum/len(pf.deathSites)))
    i += 1

d = pfs[1].motion((0., 0.), 0., 0.).d
h0_params = mot2_vals
bS = pfs[1].birthSites
dS = pfs[1].deathSites
param_step = 0.1

K = 10  # Number of iteration loops to do.
M = 1000  # Max Number of particles to compute h values.
M = min(M, len(bS), len(dS))
for k in range(K):
    z = 3**len(h0_params)
    val = np.zeros(z)
    h0_new = np.zeros((3, z))
    for j in range(z):
        zTemp = j
        for m in range(len(h0_params)):
            temp = floor(zTemp / (3**(len(h0_params)-m-1)))
            h0_new[m, j] = h0_params[m]*(1 + param_step)**(temp-1)
            zTemp = zTemp-temp*3**(len(h0_params)-1-m)
        h = hFunction(d, *h0_new[:, j])
        hBirthSum = 0.
        hDeathSum = 0.
        for p in bS[-M:]:
            hBirthSum += h.val(*p)
        for p in dS[-M:]:
            hDeathSum += h.val(*p)

        val[j] = (hBirthSum / len(bS) / (hDeathSum / len(dS)))
    if k == 0:
        val_init = val[13]

    h0_params = h0_new[:, np.argmax(val)]

print("Original Ratio of birth-h to death-h: ", val_init)
print("New Ratio of birth-h to death-h: ", np.max(val))
print("Best parameter values: ", h0_params)

print("\nTotal Time Taken: ", round(time.time()-startTime), " seconds")

if useLogger:
    sys.stdout.close()
