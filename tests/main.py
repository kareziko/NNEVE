# %%

import tensorflow as tf

# ; disable_gpu()
from nneve.common.logconfig import configure_logger
from nneve.quantum_oscilator import QOConstants, QONetwork, QOParams, QOTracker

# ; from nneve.benchmark.testing import disable_gpu


configure_logger(is_debug=True)
tf.random.set_seed(0)

# %%
nn = QONetwork(
    constants=QOConstants(
        k=4.0,
        mass=1.0,
        x_left=-6.0,
        x_right=6.0,
        fb=0.0,
        sample_size=500,
        tracker=QOTracker(),
    )
)
nn.summary()

# %%
generation_cache = []
params = QOParams(c=-2.0)

# %%
generation_cache.extend(nn.train_generations(params, 10, 1000))
