#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the article "Monte Carlo Methods for the Neutron Transport Equation.

By A. Cox, A. Kyprianou, S. Harris, M. Wang.

Thi sfile contains the code to produce the plots in the case of the 1D version
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

from NTE1D import PathsLinear, PathsCons, PathsProd, PathsMini
from simulate_tools import SimulateFunction
from Fixedpt import Eigenvalue, Eigenfunction
from scipy import stats

import time

import os
import sys
from pathlib import Path

from Logger import Logger

# Choose which plots to produce.
BranchingEstimate = False
BranchingRate = False
MemoryCosts = False
LogMemoryCosts = False
NBPvsNRW1D = True
hTransf = False

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


if BranchingEstimate:
    # =============================================================================
    # Theoretical value of Ei
    # =============================================================================

    print("~~~~ First Plot: Estimate of lambda over time via Branching ~~~~\n")
    print("(i) Super-critical case:")
    beta = 0.5  # Branching Rate, per unit time. Binary branching.
    alpha = 1.7  # Initial constant scattering rate

    T_0 = 0  # Starting time
    T = 60  # Final time
    X_0 = 0  # Initial position
    V_0 = 1  # Initial speed
    L = 1  # Physical domain [-L, L], only consider symmetric interval
    a = -L  # Lower boundary
    b = L  # Upper boundary

    # Start time for computing the weight.
    #   Must be strictly smaller than the lower bound of sample_times.
    t_start = 0

    sample_times = np.linspace(0.2 * T, 0.99 * T, 41)

    # =============================================================================
    # # Theoretical value of Ei (Supercritical)
    # =============================================================================

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\nTheoretical value of the leading eigenvalue =", y.value)
    print("\nEstimated error from solving the fixed point equation =", y.res)

    print("\nSimulate the whole system: with branching, constant scattering")

    fun = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun2 = lambda : fun(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    # =============================================================================
    # # Simulate NBP (Supercritical)
    # =============================================================================

    fun3 = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun4 = lambda : fun3(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    n_sim = 2000

    sim = SimulateFunction(fun4, n_sim, ProgBar=True)

    Ei = np.divide(np.log(np.mean(sim.z, 0)), sample_times)

    print("\nEstimate of eigenvalue at sample times: \n", Ei)

    # =============================================================================
    # # Plotting (Supercritical)
    # =============================================================================

    plt.subplot(121)

    ei, = plt.plot(sample_times, y.value*np.ones(len(sample_times)), '--')
    br, = plt.plot(sample_times, Ei, 'o-', markersize=4)

    # plt.title("Estimate of eigenvalue over time (super-critical)")
    plt.title(r"$\lambda_* =$ {0:.4f}".format(y.value))
    plt.xlabel(r"Time")
    plt.ylabel(r"Lead eigenvalue")

    # if useLogger:
    #     plt.savefig(output_dir+'/EstEigen1DSup.pdf', format='pdf')

    print("(ii) Sub-critical case:")
    beta = 0.43  # Branching Rate, per unit time. Binary branching.
    alpha = 1.7  # Initial constant scattering rate

    # =============================================================================
    # # Theoretical value of Ei (Subcritical)
    # =============================================================================

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\nTheoretical value of the leading eigenvalue =", y.value)
    print("\nEstimated error from solving the fixed point equation =", y.res)

    print("\nSimulate the whole system: with branching, constant scattering")

    fun = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun2 = lambda : fun(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    # =============================================================================
    # # Simulate NBP (Subcritical)
    # =============================================================================

    fun3 = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun4 = lambda : fun3(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    sim = SimulateFunction(fun4, n_sim, ProgBar=True)

    Ei = np.divide(np.log(np.mean(sim.z, 0)), sample_times)

    print("\nEstimate of eigenvalue at sample times: \n", Ei)

    # =============================================================================
    # # Plotting (Subcritical)
    # =============================================================================

    plt.subplot(122)

    ei, = plt.plot(sample_times, y.value*np.ones(len(sample_times)), '--')
    br, = plt.plot(sample_times, Ei, 'o-', markersize=4)

    # plt.title("Estimate of eigenvalue over time (sub-critical)")
    plt.title(r"$\lambda_* =$ {0:.4f}".format(y.value))
    plt.xlabel(r"Time")
    plt.ylabel(r"Lead eigenvalue")

    plt.tight_layout()

    if useLogger:
        plt.savefig(output_dir+'/EstEigen1D.pdf', format='pdf')


if BranchingRate:
    beta = 0.02  # Branching Rate, per unit time. Binary branching.
    alpha = 1  # Initial constant scattering rate

    T_0 = 0  # Starting time
    T = 200  # Final time
    X_0 = 0  # Initial position
    V_0 = 1  # Initial speed
    L = 10  # Physical domain [-L, L], only consider symmetric interval
    a = -L  # Lower boundary
    b = L  # Upper boundary

    t_start = 0
    # start time for computing the weight,
    # strictly smaller than the lower bound of sample_times

    sample_times = np.linspace(0.2*T, 0.99*T, 51)
    print("~~~~ Second Plot: Estimate of lambda in no of particles ~~~~\n")

    # =============================================================================
    # theoretic value of Ei
    # =============================================================================

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\n Theoretical value of the leading eigenvalue = ", y.value)
    print("\n Estimated error from solving the fixed point equation = ", y.res)

    fun_combine = lambda arg: np.divide(np.log(np.mean(arg, 0)), sample_times)

    fun_combine_var = lambda arg: np.log(np.var(arg, 0))

    Fun1 = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    Fun2 = lambda : Fun1(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    # =============================================================================
    # 100
    # =============================================================================

    n_sim = 100

    n_rep = 10  # repeat times
    E1 = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_prod1 = SimulateFunction(Fun2, n_sim, ProgBar=True)
        E1[i] = np.divide(np.log(np.mean(sim_prod1.z, 0)),
                          sample_times-t_start) - y.value
        i += 1

    e1 = np.zeros(len(sample_times))
    for j in range(len(sample_times)):
        e1[j-1] = np.mean(np.multiply(E1[:, j-1], E1[:, j-1]), 0)

    print("\n n=100: MSE at sample times: ", e1)

    # =============================================================================
    # 1000
    # =============================================================================

    n_sim = 1000

    n_rep = 10  # repeat times
    E2 = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_prod2 = SimulateFunction(Fun2, n_sim, ProgBar=True)
        E2[i] = np.divide(np.log(np.mean(sim_prod2.z, 0)),
                          sample_times-t_start) - y.value
        i += 1

    e2 = np.zeros(len(sample_times))
    for j in range(len(sample_times)):
        e2[j-1] = np.mean(np.multiply(E2[:, j-1], E2[:, j-1]), 0)

    print("\n n=1000: MSE at sample times: ", e2)

    # =============================================================================
    # 10^4
    # =============================================================================

    n_sim = 10000

    n_rep = 10  # repeat times
    E3 = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_prod3 = SimulateFunction(Fun2, n_sim, ProgBar=True)
        E3[i] = np.divide(np.log(np.mean(sim_prod3.z, 0)),
                          sample_times-t_start) - y.value
        i += 1

    e3 = np.zeros(len(sample_times))
    for j in range(len(sample_times)):
        e3[j-1] = np.mean(np.multiply(E3[:, j-1], E3[:, j-1]), 0)

    print("\n n=10000: MSE at sample times: ", e3)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log(sample_times), np.log(e3))
    print("\n Estimate of the slope: ", slope)
    print("\n R squared: ", r_value**2)

    # =============================================================================
    # Plotting
    # =============================================================================

    plt.figure()
    plt.subplot(122)

    plt.loglog(sample_times, np.exp(slope * np.log(sample_times) + intercept),
               'r--', label=r"Linear Fit (slope $\approx$ {0:.2f})".format(
                   slope))
    plt.loglog(sample_times, e3, label=r"$10^4$ Simulations")
    plt.legend()

    # plt.title(r"MSE of $10^4$ simulations (log-log scale)")
    plt.xlabel(r"log(Time)")
    plt.ylabel(r"log(MSE)")

    # if useLogger:
    #     plt.savefig(output_dir+'/MSENBPEstimate.pdf', format='pdf')

    plt.subplot(121)

    lin_plot_conv, = plt.plot(sample_times, fun_combine(sim_prod1.z),
                              label=r"$10^2$")
    branch_plot_conv, = plt.plot(sample_times, fun_combine(sim_prod2.z),
                                 label=r"$10^3$")
    mini_plot_conv, = plt.plot(sample_times, fun_combine(sim_prod3.z),
                               label=r"$10^4$")

    lambda_plot_conv, = plt.plot(sample_times, y.value*np.ones(
        np.shape(sample_times)), 'g--', label=r"$\lambda_*$")

    plt.legend(handles=[lin_plot_conv, branch_plot_conv,
                        mini_plot_conv, lambda_plot_conv])
    # plt.title(r"Estimate of eigenvalue over time")
    plt.ylabel(r"Estimated Eigenvalue")
    plt.xlabel(r"Time")
    plt.tight_layout()

    if useLogger:
        plt.savefig(output_dir+'/MSEBranching.pdf', format='pdf')

if MemoryCosts:
    print("~~~~ Third Plot: Estimate of Memory Cost ~~~~\n")

    print("(i) Super-critical case:")

    T_0 = 0
    T_1 = 50
    T = np.linspace(0.2*T_1, 0.99*T_1, 50)
    delta_t = (0.99-0.5)*T_1/100
    TC = np.append(T, T_1)
    TC = TC[1:]
    X_0 = -0.3
    V_0 = 1
    L = 1
    a = -L
    b = L
    alpha = 1
    beta = 0.75

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\n Theoretical value of the leading eigenvalue =", y.value)

    n_sim = 500
    nb_path = np.zeros((n_sim, len(T)))
    nb_alive = np.zeros((n_sim, len(T)))
    nb_new = np.zeros((n_sim, len(T)))
    nb_births = np.zeros((n_sim, len(T)))

    for k in range(n_sim):
        Z = PathsCons(T_0, T_1, X_0, V_0, beta, alpha, a, b)
        nb_path[k] = np.array([Z.count_scatters(t) for t in T])
        nb_alive[k] = np.array([Z.count(t) for t in T])
        nb_births[k] = np.array([Z.count_was_alive(t) for t in T])

    #  Estimate of phi
    x = np.linspace(-L, L, 100)
    phi = (y.varphi(X_0, V_0) * alpha / y.value *
           4 * L * np.mean(y.varphi_tilde(x, V_0)))
    print("\n Value of kappa_4", phi * y.value)

    NP = np.divide(np.log(np.mean(nb_births, 0)), T)
    # print("\n", np.mean(nb_alive, 0))

    F = lambda t: y.value+(np.log(beta)+np.log(phi))/t
    FF = np.array([F(t) for t in T])
    expf = np.array([np.exp(t*F(t)) for t in T])

    plt.figure()
    plt.subplot(121)
    fit, = plt.plot(T, expf, label='Prediction')
    path, = plt.plot(T, np.mean(nb_births, 0), 'D-', markersize=2,
                     label='Simulation')

    plt.legend(handles=[path, fit])
    plt.title(r"$\lambda_* =$ {0:.4f}".format(y.value))
    plt.xlabel('Time')

    # if useLogger:
    #     plt.savefig(output_dir+'/MemoryCostSup.pdf', format='pdf')

    print("(ii) Sub-critical case:")

    alpha = 1
    beta = 0.6

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\n Theoretical value of the leading eigenvalue = ", y.value)

    n_sim = 5000
    nb_path = np.zeros((n_sim, len(T)))
    nb_alive = np.zeros((n_sim, len(T)))
    nb_births = np.zeros((n_sim, len(T)))

    for k in range(n_sim):
        Z = PathsCons(T_0, T_1, X_0, V_0, beta, alpha, a, b)
        nb_path[k] = np.array([Z.count_scatters(t) for t in T])
        nb_alive[k] = np.array([Z.count(t) for t in T])
        nb_births[k] = np.array([Z.count_was_alive(t) for t in T])

    NP = np.divide(np.log(np.mean(nb_path, 0)), T)
    Ei = np.divide(np.log(np.mean(nb_alive, 0)), T)
    # print("\n", np.mean(nb_alive, 0))

    plt.subplot(122)
    path, = plt.plot(T, np.mean(nb_births, 0), 'D-', markersize=2,
                     label='Simulation')

    plt.title(r"$\lambda_* =$ {0:.4f}".format(y.value))
    plt.xlabel('Time')
    plt.tight_layout()

    if useLogger:
        plt.savefig(output_dir+'/MemoryCost.pdf', format='pdf')


if LogMemoryCosts:

    print("~~~~ Fourth Plot: Log Estimate of Memory, Computation Cost ~~~~\n")

    print("(i) Memory cost:")

    # =============================================================================
    # Memory Costs
    # =============================================================================

    T_0 = 0
    T_1 = 400
    T = np.linspace(0.2*T_1, 0.99*T_1, 50)
    delta_t = (0.99-0.5)*T_1/100
    TC = np.append(T, T_1)
    TC = TC[1:]
    X_0 = -0.3
    V_0 = 1
    L = 1
    a = -1
    b = 1
    alpha = 1
    beta = 0.683

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\n Theoretical value of the leading eigenvalue =", y.value)

    n_sim = 500
    # nb_path = np.zeros((n_sim, len(T)))
    # nb_alive = np.zeros((n_sim, len(T)))
    nb_births = np.zeros((n_sim, len(T)))

    for k in range(n_sim):
        Z = PathsCons(T_0, T_1, X_0, V_0, beta, alpha, a, b)
        # nb_path[k] = np.array([Z.count2(t) for t in T])
        # nb_alive[k] = np.array([Z.count(t) for t in T])
        nb_births[k] = np.array([Z.count_was_alive(t) for t in T])

    # Estimate of phi
    # Alive = np.mean(nb_alive, 0)
    # phi = np.mean([np.divide(Alive[20:-1:1], np.exp(y.value*T[20:-1:1]))])
    # print("\n Estimate of phi", phi)

    NP = np.divide(np.log(np.mean(nb_births, 0)), T)

    #  Estimate of phi
    x = np.linspace(-L, L, 100)
    phi = (y.varphi(X_0, V_0) * alpha / y.value *
           4 * L * np.mean(y.varphi_tilde(x, V_0)))
    print("\n Value of kappa_4", phi * y.value)

    # Ei = np.divide(np.log(np.mean(nb_alive, 0)), T)
    # print("\n", np.mean(nb_alive,0))

    F = lambda t: y.value+np.log(phi)/t
    FF = np.array([F(t) for t in T])

    plt.figure()
    plt.subplot(121)

    fit, = plt.plot(T, FF, label=r'Prediction')
    path, = plt.plot(T, NP, label=r"$\log(C[0, 1])/t$")

    lam, = plt.plot(T, y.value * np.ones(np.shape(T)), '--',
                    label=r"$\lambda_*$")
    plt.legend(handles=[path, fit, lam])
    plt.xlabel(r"Time")
    # plt.title("Estimation of memory cost over time")

    # if useLogger:
    #     plt.savefig(output_dir+'/MemoryCostSup.pdf', format='pdf')

    # =============================================================================
    # Processing Costs
    # =============================================================================

    print("(ii) CPU cost:")

    T_0 = 0
    T_1 = 80
    T = np.linspace(0.2*T_1, 0.99*T_1, 50)
    delta_t = (0.99-0.5)*T_1/100
    TC = np.append(T, T_1)
    TC = TC[1:]
    X_0 = -0.3
    V_0 = 1
    L = 1
    a = -1
    b = 1
    alpha = 1
    beta = 0.683

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\n Theoretical value of the leading eigenvalue =", y.value)

    n_sim = 500
    nb_scatter = np.zeros((n_sim, len(T)))

    for k in range(n_sim):
        Z = PathsCons(T_0, T_1, X_0, V_0, beta, alpha, a, b)
        nb_scatter[k] = np.array([Z.count_scatters(t) for t in T])

    x = np.linspace(-L, L, 100)
    phi = (y.varphi(X_0, V_0) * beta / y.value *
           4 * L * np.mean(y.varphi_tilde(x, V_0)))
    print("\n Value of kappa_4", phi * y.value)

    NP = np.divide(np.log(np.mean(nb_scatter, 0)), T)

    F = lambda t: y.value+(np.log(beta)+np.log(phi))/t
    FF = np.array([F(t) for t in T])

    plt.subplot(122)

    fit, = plt.plot(T, FF, label=r'Prediction')
    path, = plt.plot(T, NP, label=r'$\log C[1, 0]/t$')

    lam, = plt.plot(T, y.value * np.ones(np.shape(T)), '--',
                    label=r"$\lambda_*$")
    plt.legend(handles=[path, fit, lam])
    plt.xlabel(r"Time")
    plt.tight_layout()

    if useLogger:
        plt.savefig(output_dir+'/MemoryCPUCost.pdf', format='pdf')

    # plt.title("Estimation of computational cost over time")

if NBPvsNRW1D:
    beta = 0.48  # Branching Rate, per unit time. Binary branching.
    alpha = 1.7  # Initial constant scattering rate

    T_0 = 0  # Starting time
    T = 60   # Final time
    X_0 = 0  # Initial position
    V_0 = 1  # Initial speed
    L = 1    # Physical domain [-L, L], only consider symmetric interval
    a = -L   # Lower boundary
    b = L    # Upper boundary

    # Start time for computing the weight.
    #   Strictly smaller than the lower bound of sample_times
    t_start = 0
    sample_times = np.linspace(0.2*T, 0.99*T, 21)

    # =============================================================================
    # Theoretical value of Eigenvalue
    # =============================================================================

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\n Theoretical value of the leading eigenvalue = ", y.value)
    print("\n Simulate the whole system: with branching, constant scattering")

    # =============================================================================
    # Simulate NBP
    # =============================================================================

    print("\n Branching system")

    fun = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun2 = lambda : fun(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    n_sim = 20000

    sim = SimulateFunction(fun2, n_sim, ProgBar=True)

    Ei = np.divide(np.log(np.mean(sim.z, 0)), sample_times)

    print("\n Estimate of eigenvalue at sample times:", Ei)

    # =============================================================================
    # Vanilla many-to-one
    # =============================================================================

    print("\n Neutron Random Walk")

    fun3 = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun4 = lambda : fun3(PathsCons(T_0, T, X_0, V_0, 0, alpha, a, b))

    n_sim = 2000000

    sim = SimulateFunction(fun4, n_sim, ProgBar=True)

    Ei2 = np.divide(np.log(np.mean(sim.z, 0)), sample_times) + beta

    print("\n Estimate of eigenvalue at sample times:", Ei2)

    # =============================================================================
    # Plotting
    # =============================================================================

    # with plt.rc_context({'figure.figsize': [4., 3.2]}):
    plt.figure()

    ei, = plt.plot(sample_times, y.value*np.ones(len(sample_times)), 'g--',
                    label=r"$\lambda_*$")
    br, = plt.plot(sample_times, Ei, 'o-', markersize=4, label='NBP')
    rw, = plt.plot(sample_times, Ei2, 'D-', markersize=4, label='NRW')

    plt.legend(handles=[br, rw, ei])

    # plt.title("Estimate of eigenvalue over time")
    plt.xlabel(r"Time")
    plt.ylabel(r"Eigenvalue")
    plt.tight_layout()

    if useLogger:
        plt.savefig(output_dir+'/1DEigenEstNBPRW.pdf', format='pdf')

if hTransf:
    beta = 0.43  # Branching Rate, per unit time. Binary branching.
    alpha = 2    # Initial constant scattering rate

    T_0 = 0  # Starting time
    T = 40   # Final time
    X_0 = 0  # Initial position
    V_0 = 1  # Initial speed
    L = 1    # Physical domain [-L, L], only consider symmetric interval
    a = -L   # Lower boundary
    b = L    # Upper boundary

    # Start time for computing the weight.
    #   Should be strictly smaller than the lower bound of sample_times.
    t_start = 0
    sample_times = np.linspace(0.2*T, 0.99*T, 21)

    # =============================================================================
    # Plotting the three approximations of the eigenfunction
    # =============================================================================

    y = Eigenvalue(a, b, alpha, V_0, beta, 50)
    ti = np.linspace(-1, 1, 100, endpoint=False)
    EiF = np.array([Eigenfunction(a, b, alpha, V_0, beta, y).varphi(x)
                    for x in ti])
    Lin = np.array([Eigenfunction(a, b, alpha, V_0, beta, y).linear(x)
                    for x in ti])
    Min = np.array([Eigenfunction(a, b, alpha, V_0, beta, y).mini(x)
                    for x in ti])
    Prod = np.array([Eigenfunction(a, b, alpha, V_0, beta, y).product(x)
                     for x in ti])

    plt.figure()
    plt.subplot(121)
    F_plot, = plt.plot(ti, EiF, '-', label="Eigenfunction")

    Min_plot, = plt.plot(ti, Min, '-', label="$h_1(x)$")
    Lin_plot, = plt.plot(ti, Lin, '-', label="$h_2(x)$")
    Prod_plot, = plt.plot(ti, Prod, '-', label="$h_3(x)$")

    plt.legend(handles=[F_plot, Min_plot, Lin_plot, Prod_plot])
    # plt.title("Shape of h")
    plt.xlabel(r"$x$")

    # =============================================================================
    # Numerical Simulations: quadratic case
    # =============================================================================

    K = 20
    y = Eigenvalue(a, b, alpha, np.abs(V_0), beta, K)

    print("\nTheoretical value of the leading eigenvalue =", y.value)
    print("\nEstimated error from solving the fixed point equation =",  y.res)

    print("\nMany-to-one, Use quadratic approximation")

    Fun1 = lambda arg: np.array([np.multiply(arg.integral(t_start, arg2),
                                             np.exp(beta*(arg2-t_start)))
                                 for arg2 in sample_times])
    Fun2 = lambda : Fun1(PathsProd(T_0, T, X_0, V_0, 0, alpha, a, b))

    n_sim = 1000

    n_rep = 1  # repeat times
    E_prod = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_prod = SimulateFunction(Fun2, n_sim, ProgBar=True)
        E_prod[i] = np.divide(np.log(np.mean(sim_prod.z, 0)),
                              sample_times-t_start)-y.value
        i += 1

    e_prod = np.zeros(len(sample_times))
    for j in range(len(sample_times)):
        e_prod[j-1] = np.mean(np.multiply(E_prod[:, j-1], E_prod[:, j-1]), 0)

    # =============================================================================
    # Numerical Simulations: Piecewise Linear case
    # =============================================================================

    print("\n Many-to-one, Use piecewise linear approximation")

    Fun3 = lambda arg: np.array([np.multiply(arg.integral(t_start, arg2),
                                             np.exp(beta*(arg2-t_start)))
                                 for arg2 in sample_times])
    Fun4 = lambda : Fun3(PathsMini(T_0, T, X_0, V_0, 0, alpha, a, b))

    E_mini = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_mini = SimulateFunction(Fun4, n_sim, ProgBar=True)
        E_mini[i] = np.divide(np.log(np.mean(sim_mini.z, 0)),
                              sample_times-t_start)-y.value
        i += 1

    e_mini = np.zeros(len(sample_times))
    for j in range(len(sample_times)):
        e_mini[j-1] = np.mean(np.multiply(E_mini[:, j-1], E_mini[:, j-1]), 0)

    # =============================================================================
    # Numerical Simulations: Branching case
    # =============================================================================

    print("\nSimple branching")

    fun = lambda arg: np.array([arg.count(arg2) for arg2 in sample_times])
    fun2 = lambda : fun(PathsCons(T_0, T, X_0, V_0, beta, alpha, a, b))

    E_br = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_br = SimulateFunction(fun2, n_sim, ProgBar=True)
        E_br[i] = (np.divide(np.log(np.mean(sim_br.z, 0)), sample_times)
                   - y.value)
        i += 1

    e_br = np.zeros(len(sample_times))
    for j in range(len(sample_times)):
        e_br[j-1] = np.mean(np.multiply(E_br[:, j-1], E_br[:, j-1]), 0)

    # =============================================================================
    # Numerical Simulations: Linear case
    # =============================================================================

    print("\nMany-to-one, use linear approximation")

    fun5 = lambda arg: np.array([np.multiply(arg.integral(t_start, arg2),
                                             np.exp(beta*(arg2-t_start)))
                                 for arg2 in sample_times])
    fun6 = lambda : fun5(PathsLinear(T_0, T, X_0, V_0, 0, alpha, a, b))

    E_lin = np.zeros([n_rep, len(sample_times)])
    i = 0
    while i < n_rep:
        sim_lin = SimulateFunction(fun6, n_sim, ProgBar=True)
        E_lin[i] = np.divide(np.log(np.mean(sim_lin.z, 0)),
                             sample_times-t_start) - y.value
        i += 1

    # =============================================================================
    # Plotting the numerical results
    # =============================================================================

    fun_combine = lambda arg : np.divide(np.log(np.mean(arg, 0)), sample_times)

    plt.subplot(122)

    branch_plot_conv, = plt.plot(sample_times, fun_combine(sim_br.z), 's-',
                                 markersize=3, label=r"NBP")

    mini_plot_conv, = plt.plot(sample_times, fun_combine(sim_mini.z), 'o-',
                               markersize=3, label=r"$h_1$")
    lin_plot_conv, = plt.plot(sample_times, fun_combine(sim_lin.z), 'o-',
                              markersize=3, label=r"$h_2$")
    prod_plot_conv, = plt.plot(sample_times, fun_combine(sim_prod.z), 'D-',
                               markersize=3, label=r"$h_3$")
    lambda_plot_conv, = plt.plot(sample_times,
                                 y.value*np.ones(np.shape(sample_times)),
                                 '--', label=r"$\lambda_*$")

    plt.legend(handles=[branch_plot_conv, mini_plot_conv, lin_plot_conv,
                        prod_plot_conv, lambda_plot_conv])
    plt.xlabel(r'Time')
    plt.ylabel(r'Eigenvalue')
    plt.tight_layout()

    if useLogger:
        plt.savefig(output_dir+'/hCompare.pdf', format='pdf')


print("\nTotal Time Taken: ", round(time.time()-startTime), " seconds")

if useLogger:
    sys.stdout.close()
