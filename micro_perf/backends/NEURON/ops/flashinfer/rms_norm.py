import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.vector_norm_ops import RMSNormOp


try:
    import neuronxcc.nki.language as nl

    @ProviderRegistry.register_vendor_impl("rms_norm", "nki")
    class NKIRMSNormOp(RMSNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["nki"]

        def rms_norm_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            orig_shape = src.shape
            src = src.view(-1, src.shape[-1])

            if self.add_residual:
                residual = tensor_mapping["residual"]
                residual = residual.view(-1, residual.shape[-1])
                src = src + residual

            dst = nl.rms_norm(src, weight, self.epsilon)
            return dst.view(orig_shape)

except ImportError:
    pass
