from torch import nn
from torch import einsum
import numpy as np
import einops
import torch

class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, xs, mask=None):
        # x (batch, seq_len, dim)
        to_qkv = nn.Linear(xs.shape[-1], 3 * xs.shape[-1])

        qkv = to_qkv(xs)
        q, k, v = tuple(einops.rearrange(qkv, 'b t (d n) -> n b t d', n=3))

        # dot prod
        scaled_dot_prod = einsum('b i d, b j d -> b i j', q, k) / (q.shape[-1] ** 0.5)

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)
        attn = torch.softmax(scaled_dot_prod, dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        return out
    
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, xs, heads, mask=None):

        dim = xs.shape[-1]
        to_qkv = nn.Linear(dim, dim * heads * 3, bias=False)
        qkv = to_qkv(xs)

        # decomposition
        q, k, v = tuple(einops.rearrange('b t (d k h) -> k b h t d', k = 3, h = heads))

        # Calc score
        scaled_dot_prod = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attn = torch.softmax(scaled_dot_prod, dim = -1) # (b h i j)

        # Calc result per batch and per head h
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = einops.rearrange(out, 'b h t d -> b t (h d)')

        # final linear layer
        to_out = nn.Linear(dim * heads, dim, bias=False)
        out = to_out(out)
        return out

    def __call__(self):
        if self.stop_action_to_vlm_grad and xs[0] is not None:
            # When non-zero experts attend to expert 0, block grads into expert 0's K/V,
            # but keep grads into the querying expert (Q) intact.
            q_owner = token_owner[:, None, None, :, None]  # B,1,1,T,1
            k_owner = token_owner[:, None, None, None, :]  # B,1,1,1,S
            cross_to_expert0 = (q_owner != 0) & (k_owner == 0)

            # Apply the optional per-position "no-stop" mask to surgically exclude
            # certain VLM positions (e.g., langact) from the stop_gradient path.
            # `cross_to_expert0` flags positions that get stop_gradient; we AND with
            # NOT(no_stop) so flagged positions become "still-flowing".
            if vlm_no_stop_mask is not None:
                expert0_len = xs[0].shape[1]
                vlm_stop_mask = jnp.logical_not(vlm_no_stop_mask)  # True where stop_grad applies
                # Pad to full sequence length; non-VLM positions are irrelevant for cross_to_expert0.
                vlm_stop_full = jnp.concatenate(
                    [
                        vlm_stop_mask,
                        jnp.ones(
                            (vlm_stop_mask.shape[0], token_owner.shape[1] - expert0_len), dtype=bool
                        ),
                    ],
                    axis=1,
                )
                cross_to_expert0 = cross_to_expert0 & vlm_stop_full[:, None, None, None, :]

            expert0_len = xs[0].shape[1]
            k0 = k[:, :expert0_len, ...]
            q_i = q[:, expert0_len:, ...]
            # Build a per-key gradient-passthrough k0 used for cross-attn into expert 0.
            if vlm_no_stop_mask is not None:
                # Where no_stop=True keep gradient (k0); where False detach (stop_grad(k0)).
                no_stop_3d = vlm_no_stop_mask[..., None, None]  # B,S,1,1
                k0_for_cross = jnp.where(no_stop_3d, k0, jax.lax.stop_gradient(k0))
            else:
                k0_for_cross = jax.lax.stop_gradient(k0)
            logits0_i = jnp.einsum(
                "BTKGH,BSKH->BKGTS", q_i, k0_for_cross, preferred_element_type=jnp.float32
            )
            logits = logits.at[:, :, :, expert0_len:, :expert0_len].set(logits0_i)

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        if self.stop_action_to_vlm_grad and xs[0] is not None:
            # Detach values from expert 0 when consumed by other experts.
            probs_cross = probs * cross_to_expert0.astype(probs.dtype)
            probs_self = probs - probs_cross
            # Build a v that selectively passes/blocks gradient on a per-position basis.
            if vlm_no_stop_mask is not None:
                expert0_len = xs[0].shape[1]
                v0 = v[:, :expert0_len, ...]
                no_stop_3d = vlm_no_stop_mask[..., None, None]  # B,S_vlm,1,1
                v0_mixed = jnp.where(no_stop_3d, v0, jax.lax.stop_gradient(v0))
                v_for_cross = jnp.concatenate(
                    [v0_mixed, jax.lax.stop_gradient(v[:, expert0_len:, ...])], axis=1
                )
            else:
                v_for_cross = jax.lax.stop_gradient(v)
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs_self, v) + jnp.einsum(
                "BKGTS,BSKH->BTKGH", probs_cross, v_for_cross
            )
        else:
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")




        if self.stop_action_to_vlm_grad and xs[0] is not None:
            # When non-zero experts attend to expert 0, block grads into expert 0's K/V,
            # but keep grads into the querying expert (Q) intact.
            q_owner = token_owner[:, None, None, :, None]  # B,1,1,T,1
            k_owner = token_owner[:, None, None, None, :]  # B,1,1,1,S
            cross_to_expert0 = (q_owner != 0) & (k_owner == 0)

            expert0_len = xs[0].shape[1]
            k0 = k[:, :expert0_len, ...]
            logits0_i = jnp.einsum(
                "BTKGH,BSKH->BKGTS", q_i, jax.lax.stop_gradient(k0), preferred_element_type=jnp.float32
            )
            logits = logits.at[:, :, :, expert0_len:, :expert0_len].set(logits0_i)

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        if self.stop_action_to_vlm_grad and xs[0] is not None:
            # Detach values from expert 0 when consumed by other experts.
            probs_cross = probs * cross_to_expert0.astype(probs.dtype)
            probs_self = probs - probs_cross
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs_self, v) + jnp.einsum(
                "BKGTS,BSKH->BTKGH", probs_cross, jax.lax.stop_gradient(v)
            )
        else:
            encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")