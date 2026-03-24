import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.xccl_ops import Host2DeviceOp

OP_MAPPING = {
    "torch": Host2DeviceOp
}
