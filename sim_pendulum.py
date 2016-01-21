#!/usr/bin/env python
import numpy
import csv
import sys
import pandas

# time step
dt = numpy.float64(0.05)
# constants
M = 1.
L_0 = 1.
g = 9.81

n = 1.5
K = (n*n - 1) * M*g/L_0

# initial conditions
theta = numpy.float64(0)
L = L_0 + M*g/K

T_p = 2.*numpy.pi/numpy.sqrt(g/L)
print "Tp = {} T/dt = {}".format(T_p, T_p/dt)
T_k = 2.*numpy.pi/numpy.sqrt(K/M)
print "Tk = {} T/dt = {}".format(T_k, T_k/dt)
print "Tk/Tp = {}".format(T_k/T_p)

# 2mgl = 1/2 m l^2 w^2
w_escape = numpy.sqrt(4*g/L)

w_0 = numpy.float64(sys.argv[1])
w = w_0
#
v_l_0 = 0
v_l = v_l_0
#
t = numpy.float64(0.0)
# 
print >>sys.stderr, "w_esc = {}".format(w_escape)

theta_last = theta
data = []
t_s = []

theta += w*dt/2.
L += v_l*dt/2.

for i in range(int(sys.argv[2])*int(T_p/dt)):
    w  += -dt*g*numpy.sin(theta)/L
    v_l += -K*(L-L_0)/M + g*numpy.cos(theta) + w*w*L
    
    theta += w*dt
    theta = numpy.fmod(theta, 2.*numpy.pi)
    L += v_l*dt

    t += dt
    data.append([t, theta, w, L, v_l])
    if theta_last < 0 and theta > 0:
        t_s.append(t)
    theta_last = theta

t_s = [t_s[i] - t_s[i-1] for i in range(1,len(t_s)) ]
print "avg period = {} std periods = {}".format(numpy.average(t_s), numpy.std(t_s))
df = pandas.DataFrame().from_records(data)
df.columns = ["t", "theta", "omega", "l", "v_l"]
df.set_index("t")
ax = df.plot(kind="scatter", x="theta", y="omega", marker=".")
fig = ax.get_figure()
fig.savefig("phase1.png")
ax = df.plot(kind="scatter", x="l", y="v_l", marker=".")
fig = ax.get_figure()
fig.savefig("phase2.png")
