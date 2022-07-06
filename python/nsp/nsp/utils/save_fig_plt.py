import os
import datetime

def save_fig(pl, path, dpi=400, overwrite = False):
    if (not overwrite) and os.path.exists(path):
        print("Failed save figure : given path exsist {}".format(path))
        dt = datetime.datetime.today()  # ローカルな現在の日付と時刻を取得
        path = "_".join(str(dt).split(" ")) + ".jpeg"
        print("Instead, figure name changes : {}".format(path))
    pl.savefig(path, dpi = dpi)