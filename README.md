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

- ## v1.2
  - OS-X get some error when import torchvision. (`/home/user/miniconda3/envs/py39/lib/python3.9/site-packages/torchvision/io/image.py:13`)
  - Please just ignore that.


# environment preparation

- ## Docker
  - use docker container inside `.devcontainer`

- ## python
  - anaconda
    -
  - use pip
    - setup with `pip install -r .devcontainer/requirements.txt`
    - export requirement with `pip freeze > requirements.txt`
      - note that you need to delete `PyGObject==3.42.0` and add `--extra-index-url https://download.pytorch.org/whl/cpu` before calling torch packages.