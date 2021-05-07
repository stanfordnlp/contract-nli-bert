from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter


class SummaryWriter(object):
    def __init__(self, path: str, reduce_func=None):
        if reduce_func is None:
            reduce_func = lambda values: (sum(values) / len(values) if len(values) > 0 else None)
        self.tb_writer = _SummaryWriter(path)
        self.reduce_func = reduce_func
        self.clear()

    def __del__(self):
            self.tb_writer.close()

    def clear(self):
        self.state = defaultdict(list)

    def add_scalar(self, key, value, num: int=1):
        self.state[key].extend([value] * num)

    def write(self, global_step):
        for key, values in self.state.items():
            val = self.reduce_func(values)
            if val is not None:
                self.tb_writer.add_scalar(key, val, global_step)
        self.clear()
