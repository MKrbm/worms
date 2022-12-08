import numpy as np
import sys
sys.path.append('../../')
from nsp.utils.base_conv import *
import argparse
from nsp.utils.func import *
from nsp.utils.local2global import *

import numpy as np
import os 
import subprocess
import re

os.chdir("/home/user/project")
Js = np.arange(0.1, 2.0, 0.01)
T =  np.logspace(-1, 0.2, num=20)
print(f"J = {Js}")
for J in Js:
    for t in T:
        out = subprocess.Popen(["make", "N=1000000" ,f"J={J:.3}", "M=240", f"T={t:.5}", "SSDimerOptim"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT)
        stdout,stderr = out.communicate()
