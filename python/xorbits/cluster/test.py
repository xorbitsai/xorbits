import xorbits
import xorbits.numpy as np
import time

#exp = SLURMCluster()
#adress = exp.run()
#print(adress)
#time.sleep(5)
xorbits.init("http://c1:16379")
print(np.random.rand(5, 5).mean())