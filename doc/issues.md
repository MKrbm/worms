# 22 

# Optimize unitary matrices.
loss used for the optimization will be mel and msel ( #18 )


# Run simulation
run simulation on all of the parameters, temperature, system size. for all three types of loess : mel, msel, none

# Visualize

1. graphes which represent how well unitaries are optimized for each parameters. 
   - should be both mel and msel.
   - also, actual average sign need to be visualized (for certain fixed length)
      - 3D graph / 2D color map?
      - Also no optimization should be compared as well as mes and msel
  
2. beta vs physical quantities / system size vs physical quantities. This should be done for fixed parameters (maybe put on the same graph)
   - Should be compared for all three losses
 


About the visualizing.

1. For fixed beta and fixed system size, visualize average sign as heat map. 
  - For parameters we simulate we use two types
    1. Jz=1, J = Jx/Jz = Jy/Jz = [....], h = h/Jz = [....] 
    2. Jz=-1, as above
  - For example, beta = 1, 0.5
  - We use three type of loss, mel, msel, none
2. Then for some of the parameters from above, we're going to simulate for different system size and temperatures.
  - Change temeprature for fixed system size
  - Change system size for fixed temperature


