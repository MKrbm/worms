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
Js = np.arange(0.1, 0.5, 0.01)
T =  np.logspace(-1.6, -1, num=15)
print(T)
# print(np.logspace(-1.6, -1, num=10)[::1])

# print(f"L = 4 x 4")
# for J in Js:
#     for t in T:
#         print(f"J = {J:.3} T = {t:.3}")
#         out = subprocess.Popen(["make", "N=1000000" ,f"J={J:.3}", "M=240", f"T={t:.5}", "SSAll"], 
#                 stdout=subprocess.PIPE, 
#                 stderr=subprocess.STDOUT)
#         stdout,stderr = out.communicate()

# print("L = 8 x 8")

for J in Js:
    for t in T:
        print(f"J = {J:.3} T = {t:.3}")
        out = subprocess.Popen(["make", "N=100000" ,f"J={J:.3}", "M=240", f"T={t:.5}", "L=8", "SSAll"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT)
        stdout,stderr = out.communicate()