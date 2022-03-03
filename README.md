# Worm algorithm

- ## v4.1 
  - ~~ adding chance to worm delete at worm update step. ~~
    - ~~might not satisfy DBC.~~
  - can deal with any hermitian matrix. (not need to real-valued hamiltonian)
  - can deal with similarity-transformed hamiltonian.

- ## 4.2
  - reduce probability of zero-worm
  - warp zero-worm
  - delete worm

- ## v4.3
  - automatically reduce sign.
    - chose unit cell by ourself, then find the best local similarity transformation.7
  - make container handle cuda.
