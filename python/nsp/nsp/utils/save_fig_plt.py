import os
import datetime

def save_fig(pl, folder, name, dpi=400, overwrite = False):
    path = folder +"/" + name
    if (not overwrite) and os.path.exists(path):
        print("Failed save figure : given path exsist {}".format(path))
        dt = datetime.datetime.today()  # ローカルな現在の日付と時刻を取得
        name = "_".join(str(dt).split(" ")) + ".jpeg"
        print("Instead, figure name changes : {}".format(folder+"/" + name))
    pl.savefig(folder+"/" + name, dpi = dpi)