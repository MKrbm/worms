import numpy as np
import sys
import argparse
import numpy as np
import os 
import subprocess

os.chdir("make/Kagome")
# Js = np.arange(0.47, 0.49, 0.002)
# T =  np.logspace(-1.3, -1, num=15)
Js = np.arange(0.47, 0.5, 0.005)
# T =  np.logspace(-1.6, 0, num=20)
T =  np.logspace(0, 1, num=15)[1:]

# T = [1.1]

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

for t in T:
    print(f"T = {t:.3}")
    out = subprocess.Popen(["make", "all" ,"N=100000", "M=66","P=6", f"T={t:.5}", "L1=2", "L2=2"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    # print(stdout)