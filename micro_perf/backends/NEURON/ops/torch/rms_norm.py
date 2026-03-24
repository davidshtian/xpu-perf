import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.vector_norm_ops import RMSNormOp

OP_MAPPING = {}

class NeuronRMSNormOp(RMSNormOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

OP_MAPPING["torch"] = NeuronRMSNormOp
