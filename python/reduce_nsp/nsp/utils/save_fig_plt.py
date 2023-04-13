import os
import datetime
from matplotlib import pyplot as plt

def save_fig(fig, folder, name, dpi=400, overwrite = False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = folder +"/" + name
    if (not overwrite) and os.path.exists(path+ ".jpeg"):
        print("Failed save figure : given path exsist {}".format(path))
        dt = datetime.datetime.today()  # ローカルな現在の日付と時刻を取得
        name = name + "_" + "_".join(str(dt).split(" "))
        print("Instead, figure name changes : {}".format(folder+"/" + name))
    fig.tight_layout()
    fig.savefig(folder+"/" + name + ".jpeg", dpi = dpi)
    # plt.close(fig)