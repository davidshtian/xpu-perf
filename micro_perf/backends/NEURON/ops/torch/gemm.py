import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.tensor_gemm_ops import GemmOp

OP_MAPPING = {}

class NeuronGemmOp(GemmOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        super().vendor_parser()

        if self.dtype in ["tfloat32"]:
            raise NotImplementedError(
                "Neuron does not support tfloat32 (NVIDIA-specific)"
            )


OP_MAPPING["torch"] = NeuronGemmOp
