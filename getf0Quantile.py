import numpy as np
import glob

def getQ(a):
    f0_all = []

    for path in glob.glob("data/train/f0/2/*.npy"):
        f0 = np.load(path)

        mask = f0 > 65
        if mask.any():
            f0_all.append(f0[mask])

    f0_all = np.concatenate(f0_all)

    # print("q05:", np.quantile(f0_all, 0.05))
    # print("q10:", np.quantile(f0_all, 0.10))
    # print("q95:", np.quantile(f0_all, 0.95))
    # ratio_low = (f0_all < 120).mean()
    # print("ratio f0 < 120Hz:", ratio_low)
    if(a == 1): return np.quantile(f0_all, 0.05)
    elif(a==2): return np.quantile(f0_all, 0.10)
    else: return np.quantile(f0_all, 0.95)
