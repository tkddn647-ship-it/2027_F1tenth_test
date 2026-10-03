"""실제 모델 없이 ROS2 파이프라인만 점검: NumpyActor 를 랜덤 가중치로 바꿔 ros_node 실행."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mapless40.np_actor as npa
from mapless40.policy_spec import CONV1D_LAYERS


class RandomActor(npa.NumpyActor):
    def __init__(self, path):
        rng = np.random.default_rng(0)
        r = lambda *s: (rng.standard_normal(s) * 0.1).astype(np.float32)  # noqa: E731
        self.num_timesteps, c_in, self.convs = 0, 1, []
        for o, k, s, p in CONV1D_LAYERS:
            self.convs.append((r(o, c_in, k), r(o), s, p)); c_in = o
        self.pool_n = 9
        self.fc = (r(48, c_in * 9), r(48))
        self.st = (r(32, 17), r(32))
        self.mlp = [(r(256, 4 * 48 + 32), r(256)), (r(256, 256), r(256))]
        self.mu = (r(2, 256), r(2))


npa.NumpyActor = RandomActor
from mapless40 import ros_node  # noqa: E402
sys.argv += ["--ros-args", "-p", "model:=dummy.zip"] if "--ros-args" not in sys.argv else []
ros_node.main()
