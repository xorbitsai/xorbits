import xorbits
import xorbits.numpy as np

# exp = SLURMCluster()
# address = exp.run()
# print(address)
# time.sleep(5)
xorbits.init("http://c1:16379")
print(np.random.rand(5, 5).mean())
