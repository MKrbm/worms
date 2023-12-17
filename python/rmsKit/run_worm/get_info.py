
import sys
import os
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.path as path_utils # noqa: E402


file_path = Path(__file__)
rmsKit_directory = path_utils.extract_to_rmskit(file_path)
os.chdir(rmsKit_directory)


path = Path("array/torch/BLBQ1D_loc/J0_3_J1_1_hx_0_hz_0/1_mel/")
print("search path: {}".format(path.resolve().as_posix()))
print(path_utils.get_worm_path(path))

# check if given path exists
if not os.path.isdir(path):
    logging.debug("current dir is: ", os.getcwd())
    raise ValueError("given serch path doesn't exit.{}".format(path.resolve().as_posix()))

info_txt_files = path_utils.find_info_txt_files(path)

for file in info_txt_files:
    info = path_utils.extract_info_from_txt(file)

symb = Path("/Users/keisukemurota/Documents/todo/worms/job/link/BLBQ1D/J0_-3.000_J1_1.000_hz_0.000_hx_0.000_lt_1/")
print("search path: {}".format(symb.resolve().as_posix()))


