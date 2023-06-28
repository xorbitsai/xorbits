import xorbits.pandas as pd
from xorbits._mars import options
# options.show_progress=False
df = pd.DataFrame(pd.read_feather("../ds.feather").execute().fetch(), chunk_size=500)[500:1000]

res = df.dedup().execute()

import pdb; pdb.set_trace()

