import numpy as np
import sys
sys.path.append('../../')


import numpy as np
import os 
import subprocess
import re

Js = np.arange(0.1, 1.2, 0.01)


for J in Js:
    print(J)
    out = subprocess.Popen(["python", "SS_exact.py","-J", f"{J:.3}", "-n", "200", "-m", "60"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    print(stdout)