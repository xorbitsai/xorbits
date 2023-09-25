import xorbits
import xorbits.numpy as np
from .Slurm import SLURMCluster
import time

exp = SLURMCluster()
adress = exp.run()
print(adress)
time.sleep(5)
xorbits.init("http://c1:16379")
print(np.random.rand(100, 100).mean())