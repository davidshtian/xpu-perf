import sys
import pathlib
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp

OP_MAPPING = {}


try:
    from neuronxcc.nki.kernels.attention import flash_fwd, FlashConfig

    @ProviderRegistry.register_vendor_impl("flash_attention", "nki")
    class NKIFlashAttentionOp(FlashAttentionOp):
        """Flash Attention using NKI flash_fwd kernel on AWS Neuron.

        NKI flash_fwd expects:
          q: (bs, n_heads, d, seq_q)
          k: (bs, nk_heads, d, seq_k)
          v: (bs, nv_heads, d, seq_v)  when should_transpose_v=True
          o: (bs, n_heads, seq_q, d)
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.attn_mode == "prefill":
                if not (
                    self.dtype == "bfloat16"
                    and self.dst_dtype == "bfloat16"
                    and self.cache_dtype == "bfloat16"
                    and self.qk_compute_dtype == "bfloat16"
                    and self.pv_compute_dtype == "bfloat16"
                    and self.cache_type == "linear"
                ):
                    raise ValueError(
                        f"{type(self).__name__} prefill only supports bfloat16 with linear cache."
                    )

                if self.batch_size != 1:
                    raise ValueError(
                        f"{type(self).__name__} prefill only supports batch_size == 1."
                    )

                if self.cache_lens[0] != 0:
                    raise ValueError(
                        f"{type(self).__name__} prefill only supports cache_lens[0] == 0."
                    )

            elif self.attn_mode == "decode":
                raise ValueError(
                    f"{type(self).__name__} does not support decode mode."
                )
            else:
                raise ValueError(
                    f"{type(self).__name__} does not support attn_mode: {self.attn_mode}."
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
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # q: (num_tokens, q_head_num, head_dim) -> (1, q_head_num, head_dim, seq_q)
            q_nki = q.view(
                self.batch_size, self.num_tokens, self.q_head_num, self.head_dim
            ).permute(0, 2, 3, 1).contiguous()

            # k_cache: (bs, kv_head_num, max_kv_len, head_dim) -> (bs, kv_head_num, head_dim, seq_k)
            k_nki = k_cache[:, :, :self.num_tokens, :].permute(0, 1, 3, 2).contiguous()

            # v_cache: same transpose as k
            v_nki = v_cache[:, :, :self.num_tokens, :].permute(0, 1, 3, 2).contiguous()

            o_nki = flash_fwd[self.batch_size, self.q_head_num](
                q_nki,
                k_nki,
                v_nki,
                seed=None,
                use_causal_mask=self.is_causal,
                mixed_precision=True,
                softmax_scale=self.softmax_scale,
                config=self._flash_config,
            )

            return o_nki

    OP_MAPPING["nki"] = NKIFlashAttentionOp

except ImportError:
    pass
