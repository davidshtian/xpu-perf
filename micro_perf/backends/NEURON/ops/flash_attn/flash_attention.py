import sys
import pathlib
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp


try:
    from neuronxcc.nki.kernels.attention import flash_fwd, FlashConfig

    @ProviderRegistry.register_vendor_impl("flash_attention", "nki")
    class NKIFlashAttentionOp(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.attn_mode == "prefill":
                if self.dtype == "bfloat16" \
                    and self.dst_dtype == "bfloat16" \
                    and self.cache_dtype == "bfloat16" \
                    and self.qk_compute_dtype == "bfloat16" \
                    and self.pv_compute_dtype == "bfloat16" \
                    and self.cache_type == "linear":
                    pass
                else:
                    raise ValueError(
                        f"{type(self).__name__} prefill not support this combination."
                    )

                if self.batch_size != 1:
                    raise ValueError(
                        f"{type(self).__name__} prefill only support batch_size == 1."
                    )

                if self.cache_lens[0] != 0:
                    raise ValueError(
                        f"{type(self).__name__} prefill only support cache_lens[0] == 0."
                    )

            elif self.attn_mode == "decode":
                raise ValueError(
                    f"{type(self).__name__} does not support decode mode."
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} not support this attn_mode: {self.attn_mode}."
                )

        def vendor_impl(self):
            super().vendor_impl()

            seq_len = self.num_tokens
            if seq_len >= 2048:
                seq_tile_size = 2048
            else:
                seq_tile_size = seq_len

            self._flash_config = FlashConfig(
                seq_tile_size=seq_tile_size,
                training=False,
                should_transpose_v=True,
            )

            if self.attn_mode == "prefill":
                self._run_func = self.prefill_run

        def prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"].view(
                self.batch_size, self.num_tokens, self.q_head_num, self.head_dim
            ).permute(0, 2, 3, 1).contiguous()

            k_cache = tensor_mapping["k_cache"][:, :, :self.num_tokens, :].permute(0, 1, 3, 2).contiguous()
            v_cache = tensor_mapping["v_cache"][:, :, :self.num_tokens, :].permute(0, 1, 3, 2).contiguous()

            out = flash_fwd[self.batch_size, self.q_head_num](
                q,
                k_cache,
                v_cache,
                seed=None,
                use_causal_mask=self.is_causal,
                mixed_precision=True,
                softmax_scale=self.softmax_scale,
                config=self._flash_config,
            )
            return out

except ImportError:
    pass
