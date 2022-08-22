#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:56:22 2021

@author: geneyang
"""

import pandas as pd
import numpy as np
from scipy import optimize
import math
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve
from sklearn.metrics import r2_score
from sklearn import svm


attdata = pd.read_csv("/Users/geneyang/Documents/Arterial Travel Time/arterial travel times.csv")
attdata = attdata.drop('End time (sec)',1)
attdata = attdata.drop('v_id',1)
attdata.dtypes
attdata = attdata.sort_values(['Start time (sec)'])

cbvtestingdataset = pd.read_csv("/Users/geneyang/Documents/Arterial Travel Time/CBV.csv")

## scale X
t_start = attdata['Start time (sec)'].min()
attdata['Start time (sec)'] = attdata['Start time (sec)']-t_start
attdata.reset_index(drop=True, inplace=True)
maxdiff = attdata['Travel time (sec)'].max()

print(attdata)
startTime = attdata['Start time (sec)'].to_list()
travelTime = attdata['Travel time (sec)'].to_list()

plt.figure()
plt.plot(startTime, travelTime,'-*')
# attdata.plot(x='Start time (sec)', y='Travel time (sec)')


#%% break the data into cycles
threshold_1 = 10 # to break the data and find circles
threshold_2 = 100 #unused

cycles = [] #index of new cycles, each cycle is a pair of (start, end) inclusive
startindex = 0
endindex = 0
for i in range(1, len(travelTime)): # indexes of cycle borders
    if travelTime[i] - travelTime[i-1] > threshold_1:
        print(travelTime[i], " jumps from ", travelTime[i-1], ' by ', travelTime[i] - travelTime[i-1])
        endindex = i - 1
        if endindex - startindex >= 2:  # we need at least 2 points in one cycle to fit a line
            cycles.append((startindex, endindex))
        startindex = i
cycles.append((startindex, len(travelTime)-1)) # closing off the last cycle

## create a stack, so that we can do "last in and first out" operation; why: I will later add new circle dynamically and process the newly added circle
cyclesASstack = [cycles[i] for i in range(len(cycles)-1, -1, -1)]
cycles = cyclesASstack

print(cycles)


#%% cycle svm fitting
#according to the paper, each point has x value vehicle's delay - previous delay, y is arrival time - previous arrival time

dT = cbvtestingdataset['dT'].to_numpy()
dD = cbvtestingdataset['dD'].to_numpy()
training_X = np.column_stack((dT,dD))

training_Y = cbvtestingdataset['CBV?'].to_numpy()

clf = svm.SVC(kernel='linear', C=1.0)

clf.fit(training_X, training_Y)

w=clf.coef_[0]

a = -w[0]/w[1]

XX = np.linspace(0,10)

yy = a * XX - clf.intercept_[0] / w[1]


#%% define function fitting two lines
def fit_2line(para):  # x0 is a_1, x1 is b_1, x2 is a_2, x3 is b_2, and x4 is m
    m = math.floor(para[4])
    if m < 2 or m > R - 1:
        return 1000000000000  # large enough that it will never be the min
    sum1 = ((para[0] * t[:m] + para[1] - d[:m]) ** 2).sum()
    sum2 = ((para[2] * t[m:] + para[3] - d[m:]) ** 2).sum()
    return sum1 + sum2


def Fcontr(para):  # x0 is a_1, x1 is b_1, x2 is a_2, x3 is b_2, and x4 is m
    m = math.floor(para[4])
    boundINgroup = (1 - theta) * t[m - 1] + theta * t[m]
    return para[0] * boundINgroup + para[1] - (para[2] * boundINgroup + para[3])


constr = {'type': 'eq', 'fun': Fcontr}


#%% main function for fitting all circle and visualization

# cycles = [cycles[0]] # test: pull out one circle for test; i am testing on the 3rd circle

plt.figure()
plt.scatter(startTime, travelTime, marker='x', s=15, color='darkblue', alpha=0.7)

theta = 0.5
# cycle = (82, 84) # (28,35) # test
pre_cycle = None

# cycle = cycles[4] # for test

while len(cycles):
    cycle = cycles.pop()
    t, d = np.array(startTime[cycle[0]: cycle[1]+1]), np.array(travelTime[cycle[0]: cycle[1]+1])
    R = len(t)
    #print(t), print(d)

    leftBoundOUTgroup = startTime[0] if cycle[0] - 1 < 0 else int((1 - theta) * startTime[cycle[0] - 1] + theta * startTime[cycle[0]])
    rightBoundOUTgroup = startTime[-1] if cycle[1] + 1 >= len(startTime) else int((1 - theta) * startTime[cycle[1]] + theta * startTime[cycle[1]+1])

    if R > 2:
        ## fit with two lines
        # Initial values. It is reasonable to set -x+40000 for the linear equations,
        # and assume that the two lines are cut somewhere between the first and last point: m
        a1, b1, a2, b2, m = -1, 40000, -1, 40000, 1 + int(R/2)
        para = np.array([a1, b1, a2, b2, m])
        result2line = optimize.minimize(fit_2line, para, constraints=constr)
        a1, b1, a2, b2, m = result2line.x
        m = int(m)

        error2lines = result2line.fun
        predicted2line = list(a1 * t[:m] + b1)
        predicted2line.extend(list(a2 * t[m:] + b2))
        R2for2lines = r2_score(d, predicted2line)
    else:
        R2for2lines = -1

    ## fit t, d with one line
    result1line = LinearRegression().fit(t.reshape(-1, 1), np.array(d))
    error1line = ((d - result1line.predict(t.reshape(-1, 1)))**2).sum()
    R2for1line = result1line.score(t.reshape(-1, 1), np.array(d))#R^2 score: r2_score(d, result1line.predict(t.reshape(-1, 1)))
    a, b = result1line.coef_[0], result1line.intercept_

    # if error2lines > error1line:
    if R<=2 or (R2for2lines - R2for1line<=0.001 and (t[-1]-t[0]) <= threshold_2): # instead of comparing func errors, I compare R^2, an indicator of good fitness of a model
        #print("For cycle " + str(cycle) + " of length " + str(R) + ", we have 1 line: ")
        #print("a: ", a, " b: ", b)
        if a > 0: a, b = 0, 20
        rightBoundOUTgroup = min(int(fsolve(lambda ti: a * ti + b - 20, rightBoundOUTgroup)), rightBoundOUTgroup)
        plt.plot(np.array(range(leftBoundOUTgroup, rightBoundOUTgroup)),
                 a * np.array(range(leftBoundOUTgroup, rightBoundOUTgroup)) + b, 'b-', linewidth=1, alpha=0.7)
        plt.vlines(leftBoundOUTgroup, ymin=20, ymax=leftBoundOUTgroup * a + b, colors='darkblue', linewidth=1, alpha=0.7)
    else:
        #print("For cycle ", cycle, " of length ", R, "we have 2 lines: ")
        #print("a1: ", a1, " b1: ", b1, " a2: ", a2, " b2: ", b2, " m: ", m)
        if a1 > 0: a1, b1 = 0, 20
        if a2 > 0: a2, b2 = 0, 20
        boundINgroup = int((1 - theta) * t[m - 1] + theta * t[m]) # for plotting the break of the two lines in the group
        if (boundINgroup - t[0] > threshold_2) or (t[-1] - boundINgroup) > threshold_2: # if a circle is too long, we break it again
            cycles.append((cycle[0]+m, cycle[1]))
            cycles.append((cycle[0], cycle[0]+m-1))
            continue
        boundINgroup = min(int(fsolve(lambda ti: a1 * ti + b1 - 20, rightBoundOUTgroup)), boundINgroup) # stop at where it intersects y=20
        rightBoundOUTgroup = min(int(fsolve(lambda ti: a2 * ti + b2 - 20, rightBoundOUTgroup)), rightBoundOUTgroup) # stop at where it intersects y=20
        t1, t2 = np.array(range(leftBoundOUTgroup, boundINgroup)), np.array(range(boundINgroup, rightBoundOUTgroup))
        #print("lengths of two cycles: ", (len(t1), len(t2)))

        plt.plot(t1, a1 * t1 + b1, 'b-', linewidth=1, alpha=0.7)
        plt.plot(t2, a2 * t2 + b2, 'b-', linewidth=1, alpha=0.7)

        ## vertical line on the left breaking groups
        plt.vlines(leftBoundOUTgroup, ymin= 20, ymax=leftBoundOUTgroup*a1+b1, linewidth=1, colors='darkblue', alpha=0.7)
        ## vertical line in the middle breaking the two fitted lines
        plt.vlines(boundINgroup, ymin=20, ymax=boundINgroup * a1 + b1, linewidth=1, colors='darkblue', alpha=0.7)

    # pre_cycle = cycle

plt.ylim([0,maxdiff+5])
plt.hlines(y=20, xmin=startTime[0], xmax=startTime[-1], linewidth=1, colors='darkblue', alpha=0.7)
plt.ylabel('Travel time (sec)')
plt.xlabel('Start time (scaled; sec)')


    
