import os
from data import srdata


class RT_GENE(srdata.SRData):
    def __init__(self, args, name='RT_GENE', train=True, benchmark=False):
        super(RT_GENE, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(RT_GENE, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR/x4')

