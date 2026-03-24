import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.xccl_ops import AllReduceOp

OP_MAPPING = {
    "torch": AllReduceOp
}
