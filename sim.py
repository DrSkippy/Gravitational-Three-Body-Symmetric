#!/usr/bin/env python
import numpy
import csv
import sys
# csv output
wrtr = csv.writer(sys.stdout)
# time step
dt = numpy.float64(0.15)
# constants
G = 4.302e-3 # pc Msun-1 (km/s)2
M = 1.0 # mass sun 
mu = 1.e-8 # m = mu * M
# initial conditions
r = numpy.float64(0.5) # pc
z = numpy.float64(0.0) # pc
y = numpy.float64(0.0) # pc
theta = numpy.float(0.0) # radians

v_0 = numpy.float64(sys.argv[1])
#
tmp = numpy.power(r*r + (z-y)*(z-y), -1.5)
w = numpy.sqrt(G*M*(0.5/(r*r) + mu*r*tmp)/r)
vz = numpy.float64(v_0)
vy = numpy.float64(0.0)
vr = numpy.float64(0.0)
# header
wrtr.writerow(["t","r","vr","z","vz","y", "vy", "theta","w"]) 
#
t = numpy.float64(0.0)
for i in range(int(sys.argv[2])*int(2.0*numpy.pi/(w*dt))):
    tmp = numpy.power(r*r + (z-y)*(z-y), -1.5)
    w  += -dt*2.0*vr*w/r
    vr +=  dt*(r*w*w - G*M*(0.5/(r*r) + mu*r*tmp))
    vz += -dt*2.0*G*M*(z-y)*tmp
    vy += -dt*0.5*mu*vz 
    
    theta += w*dt
    theta = numpy.fmod(theta, 2.*numpy.pi)
    z += vz * dt
    y += vy * dt
    r += vr * dt
    t += dt
    wrtr.writerow([t, r, vr, z, vz, y, vy, theta, w]) 
