# This is readme of python sub-project for reduce negative sign project.

## How to use this?
- this python package is used with main worm algorithm code written in C++.

##
- Install jax with 
  - For gpu (currently only available for cuda gpu)
    ```Python
    pip install --upgrade pip
    # Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```
  - For cpu
    ```Python
    pip install --upgrade "jax[cpu]"
    ```
    Or you can build from source : https://jax.readthedocs.io/en/latest/developer.html#building-from-source

  - If you asked for `/usr/lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.29 not found` Then you need to update libstdc++6. 
    ```bash
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test # Ignore if not ubuntu
    sudo apt-get update
    sudo apt-get upgrade libstdc++6
    ```

    - You can check if it's installed or not via 
      `strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX`

