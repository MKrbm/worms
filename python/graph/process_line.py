import numpy as np
import pandas as pd
import re
import sys


def process_line_SS(stdout):
    energy = []
    cq = []
    susc = []
    sign = []
    time = []
    dimer = []
    weight_r = []
    
    for line in stdout.split("\n"):
        pat = "Total Energy"
        if pat in line:
            line = line.replace(pat,"")
            line = line.replace(" ","")
            line = line.replace("=","")
            line = line.split("+-")
            energy = [float(line[0]), float(line[1])]

        pat = "specific heat"
        if pat in line:
            line = line.replace(pat,"")
            line = line.replace(" ","")
            line = line.replace("=","")
            line = line.split("+-")
            cq = [float(line[0]), float(line[1])]
        
        pat = "susceptibility"
        if pat in line:
            line = line.replace(pat,"")
            line = line.replace(" ","")
            line = line.replace("=","")
            line = line.split("+-")
            susc = [float(line[0]), float(line[1])]

        pat = "average sign"
        if pat in line:
            line = line.replace(pat,"")
            line = line.replace(" ","")
            line = line.replace("=","")
            line = line.split("+-")
            sign = [float(line[0]), float(line[1])]

        pat = "Elapsed time"
        if pat in line:
            line = line.replace(pat,"")
            line = line.replace(" ","")
            line = line.replace("=","")
            line = line.replace("sec","")
            time = [float(line)]
            

            
    return energy, cq, susc, sign, time
