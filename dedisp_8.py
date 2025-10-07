
from builtins import zip
from builtins import range
import os

def myexecute(cmd):
    print("'%s'"%cmd)
    os.system(cmd)

# By default, do not output subbands
outsubs = False

nsub = 32

basename = '8'
rawfiles = '8'

# dDM steps from DDplan.py
dDMs        = [0.5, 1.0, 3.0]
# dsubDM steps
dsubDMs     = [13.0, 26.0, 78.0]
# downsample factors
downsamps   = [np.int64(4), np.int64(8), np.int64(16)]
# number of calls per set of subbands
subcalls    = [21, 10, 1]
# The low DM for each set of DMs
startDMs    = [0.0, 273.0, 533.0]
# DMs/call
dmspercalls = [26, 26, 26]


# Loop over the DDplan plans
for dDM, dsubDM, dmspercall, downsamp, subcall, startDM in zip(dDMs, dsubDMs, dmspercalls, downsamps, subcalls, startDMs):
    # Loop over the number of calls
    for ii in range(subcall):
        subDM = startDM + (ii+0.5)*dsubDM
        loDM = startDM + ii*dsubDM
        if outsubs:
            # Get our downsampling right
            subdownsamp = downsamp // 2
            datdownsamp = 2
            if downsamp < 2: subdownsamp = datdownsamp = 1
            # First create the subbands
            myexecute("prepsubband -sub -subdm %.2f -nsub %d -downsamp %d -o %s %s" %
                      (subDM, nsub, subdownsamp, basename, rawfiles))
            # And now create the time series
            subnames = basename+"_DM%.2f.sub[0-9]*"%subDM
            myexecute("prepsubband -lodm %.2f -dmstep %.2f -numdms %d -downsamp %d -o %s %s" %
                      (loDM, dDM, dmspercall, datdownsamp, basename, subnames))
        else:
            myexecute("prepsubband -nsub %d -lodm %.2f -dmstep %.2f -numdms %d -downsamp %d -o %s %s" %
                      (nsub, loDM, dDM, dmspercall, downsamp, basename, rawfiles))
