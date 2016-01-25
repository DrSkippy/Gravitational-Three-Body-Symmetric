#!/usr/bin/env python
import csv
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# arg 1 = w init
# arg 2 = n periods
# arg 3 = n ratio

# time step
dt = np.float64(0.00010)

# constants
L_0 = np.float64(1.0)  # unstretched length
g = np.float64(9.81)   # gravitation
n = np.float64(sys.argv[3])
K_over_M = (n*n - 1)*g/L_0

# initial conditions
theta = np.float64(0)
L = L_0 + g/K_over_M  # equilibrium length with gravity

# 2mgl = 1/2 m l^2 w^2
w_sep = np.sqrt(4.*g/L)
w_0 = np.float64(sys.argv[1])
w = w_0
#
v_l_0 = 0
v_l = v_l_0

# periods
T_p = 2.*np.pi/np.sqrt(g/L)
T_k = 2.*np.pi/np.sqrt(K_over_M)
# record some stuff
print "Tp = {} T/dt = {}".format(T_p, T_p/dt)
print "Tk = {} T/dt = {}".format(T_k, T_k/dt)
print "Tk/Tp = {}".format(T_k/T_p)
print "w_esc = {}".format(w_sep)

t = np.float64(0.0)
theta_last = theta

# keep some records
data = []
t_s = []

theta += w*dt/2.
L += v_l*dt/2.

for i in range(int(sys.argv[2])*int(T_p/dt)):
    w  += -dt*g*np.sin(theta)/L
    v_l += -K_over_M*(L-L_0) + g*np.cos(theta) + w*w*L
    
    theta += w*dt
    theta = np.fmod(theta, 2.*np.pi)
    L += v_l*dt

    t += dt
    data.append([t, theta, w, L, v_l])
    if theta_last < 0 and theta > 0:
        t_s.append(t)
    theta_last = theta

# periods by measure
t_s = [t_s[i] - t_s[i-1] for i in range(1,len(t_s)) ]
print "avg period = {} std periods = {}".format(np.average(t_s), np.std(t_s))

# plots
df = pd.DataFrame().from_records(data)
df.columns = ["t", "theta", "omega", "l", "v_l"]
df.set_index("t")

ax = df.plot(kind="scatter", x="theta", y="omega", marker=".")
fig = ax.get_figure()
fig.savefig("phase1.png")

ax = df.plot(kind="scatter", x="l", y="v_l", marker=".")
fig = ax.get_figure()
fig.savefig("phase2.png")

# config space
df["y_c"] = -df["l"]
df["x_c"] = df["l"] * np.sin(df["theta"])
ax = df.plot(kind="scatter", x="x_c", y="y_c", marker=".")
fig = ax.get_figure()
fig.savefig("config.png")
