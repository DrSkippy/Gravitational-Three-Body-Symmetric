#!/usr/bin/env python
import numpy
import csv
import sys
# time step
dt = numpy.float64(0.05)
# constants
G = 4.302e-3 # pc Msun-1 (km/s)2
M = 1.0 # mass sun 
mu = 1.e-8 # m = mu * M
# initial conditions
r = numpy.float64(0.5) # pc
z = numpy.float64(0.0) # pc
y = numpy.float64(0.0) # pc
theta = numpy.float(0.0) # radians

v_escape = numpy.sqrt(4*G*M/r)
v_0 = numpy.float64(sys.argv[1])
#
tmp = numpy.power(r*r + (z-y)*(z-y), -1.5)
w = numpy.sqrt(G*M*(0.5/(r*r) + mu*r*tmp)/r)
vz = numpy.float64(v_0)
vy = numpy.float64(0.0)
vr = numpy.float64(0.0)
#
t = numpy.float64(0.0)

last_z = z
last_t = t

# 
print >>sys.stderr, "v_esc = {}".format(v_escape)
print >>sys.stderr, "vz_0 = {}".format(vz)
print >>sys.stderr, "year_0 = {}".format(2.0*numpy.pi/w)
print >>sys.stderr, "comet_period (small z) = {}".format(2.0*numpy.pi*numpy.power((numpy.power(r,3)/(2.*G*M)),0.5))

with open("./returnmap.csv","wb") as f:
    # csv output
    wrtr = csv.writer(sys.stdout)
    rwrtr = csv.writer(f)
    # header
    wrtr.writerow(["t","r","vr","z","vz","y", "vy", "theta","w"]) 
    rwrtr.writerow(["t","r","vr","z","vz","y", "vy", "theta","w","period"]) 
    
    theta += w*dt/.2
    z += vz*dt/2.
    y += vy*dt/2.
    r += vr*dt/2.

    for i in range(int(sys.argv[2])*int(2.0*numpy.pi/(w*dt))):
        tmp = numpy.power(r*r + (z-y)*(z-y), -1.5)
        w  += -dt*2.0*vr*w/r
        vr +=  dt*(r*w*w - G*M*(0.5/(r*r) + mu*r*tmp))
        vz += -dt*2.0*G*M*(z-y)*tmp
        vy += -dt*0.5*mu*vz 
        
        theta += w*dt
        z += vz*dt
        y += vy*dt
        r += vr*dt
        # dissappointment?
        if min([z, last_z]) < 0 <= max([z, last_z]):
            # comet appeared this year
            # see it in either direction, so call this the half-period?
            rwrtr.writerow([t, r, vr, z, vz, y, vy, theta, w, t-last_t])
            last_t = t
        last_z = z
        theta = numpy.fmod(theta, 2.*numpy.pi)

        t += dt
        wrtr.writerow([t, r, vr, z, vz, y, vy, theta, w]) 
