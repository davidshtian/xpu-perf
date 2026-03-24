import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.xccl_ops import AllGatherOp


class NeuronAllGatherOp(AllGatherOp):
    """Override all_gather_run to use xm.all_gather instead of
    dist.all_gather_into_tensor which is incompatible with XLA backend."""

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._run_func = self.neuron_all_gather_run

    def neuron_all_gather_run(self, tensor_mapping):
        import torch_xla.core.xla_model as xm
        src = tensor_mapping["src"]
        dst = xm.all_gather(src, dim=0)
        return dst


OP_MAPPING = {
    "torch": NeuronAllGatherOp
}
