"""Vision transformer autoencoder implementation."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from transformers import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
)
from typeguard import typechecked

from beast.models.base import BaseLightningModel

def _find_prefix(keys, needle):
    """Return the prefix (everything before `needle`) for the first match."""
    for k in keys:
        i = k.find(needle)
        if i != -1:
            return k[:i]
    return ""  # no prefix

def convert_timm_mae_to_hf(
    ckpt_path: str,
    fuse_decoder=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safe load (PyTorch 2.6+)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    sd = raw.get("model", raw.get("state_dict", raw))

    hf_decoder_sd = {k: v for k, v in sd.items() if "decoder" in k}

    hf_decoder_sd.update({"mask_token": sd.get("mask_token", None)})

    # Remove DDP prefix
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    from collections import OrderedDict
    hf_sd = OrderedDict()

    def maybe(src, dst):
        if src in sd:
            hf_sd[dst] = sd[src]

    # Embeddings
    maybe("cls_token", "embeddings.cls_token")
    maybe("pos_embed", "embeddings.position_embeddings")
    maybe("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight")
    maybe("patch_embed.proj.bias",   "embeddings.patch_embeddings.projection.bias")

    # Encoder layers
    i = 0
    while True:
        prefix = f"blocks.{i}."
        if not any(k.startswith(prefix) for k in sd):
            break

        # Layernorm 1 -> layernorm_before
        maybe(f"{prefix}norm1.weight", f"encoder.layer.{i}.layernorm_before.weight")
        maybe(f"{prefix}norm1.bias",   f"encoder.layer.{i}.layernorm_before.bias")

        # Layernorm 2 -> layernorm_after
        maybe(f"{prefix}norm2.weight", f"encoder.layer.{i}.layernorm_after.weight")
        maybe(f"{prefix}norm2.bias",   f"encoder.layer.{i}.layernorm_after.bias")

        # qkv split
        qkv_w = f"{prefix}attn.qkv.weight"
        qkv_b = f"{prefix}attn.qkv.bias"

        if qkv_w in sd:
            W = sd[qkv_w]
            D3, Din = W.shape
            assert D3 % 3 == 0, "qkv weight dim mismatch"
            D = D3 // 3

            hf_sd[f"encoder.layer.{i}.attention.attention.query.weight"] = W[:D]
            hf_sd[f"encoder.layer.{i}.attention.attention.key.weight"]   = W[D:2*D]
            hf_sd[f"encoder.layer.{i}.attention.attention.value.weight"] = W[2*D:]

        if qkv_b in sd:
            b = sd[qkv_b]
            D3 = b.shape[0]
            assert D3 % 3 == 0, "qkv bias dim mismatch"
            D = D3 // 3

            hf_sd[f"encoder.layer.{i}.attention.attention.query.bias"] = b[:D]
            hf_sd[f"encoder.layer.{i}.attention.attention.key.bias"]   = b[D:2*D]
            hf_sd[f"encoder.layer.{i}.attention.attention.value.bias"] = b[2*D:]

        # Proj
        maybe(f"{prefix}attn.proj.weight",
              f"encoder.layer.{i}.attention.output.dense.weight")
        maybe(f"{prefix}attn.proj.bias",
              f"encoder.layer.{i}.attention.output.dense.bias")

        # MLP
        maybe(f"{prefix}mlp.fc1.weight", f"encoder.layer.{i}.intermediate.dense.weight")
        maybe(f"{prefix}mlp.fc1.bias",   f"encoder.layer.{i}.intermediate.dense.bias")
        maybe(f"{prefix}mlp.fc2.weight", f"encoder.layer.{i}.output.dense.weight")
        maybe(f"{prefix}mlp.fc2.bias",   f"encoder.layer.{i}.output.dense.bias")

        i += 1

    # Final encoder norm
    maybe("norm.weight", "layernorm.weight")
    maybe("norm.bias",   "layernorm.bias")

    # Return inferred model parameters (important!)
    config_overrides = {
        "num_hidden_layers": i,
        "hidden_size": hf_sd["embeddings.cls_token"].shape[-1],
    }

    return hf_sd, hf_decoder_sd, config_overrides


@torch.no_grad()
def map_timm_decoder_to_hf(hf_model, timm_sd: dict) -> dict:

    out = {}

    # Autodetect prefixes
    t_keys = list(timm_sd.keys())
    h_keys = list(hf_model.state_dict().keys())

    t_pref = _find_prefix(t_keys, "decoder_blocks.0")  # e.g. "" or "module."
    h_block_pref = _find_prefix(h_keys, "decoder_layers.0")  # e.g. "vit.decoder."
    h_dec_pref = _find_prefix(h_keys, "decoder_embed.weight") or _find_prefix(h_keys, "decoder.decoder_embed.weight")
    # Backups: many HF builds keep everything under the same prefix
    if not h_dec_pref:
        h_dec_pref = h_block_pref

    # Helper to place keys
    def hk(suffix):  # HF key
        return f"{h_dec_pref}{suffix}" if (h_dec_pref and not suffix.startswith("decoder_layers")) else f"{h_block_pref}{suffix}"

    # 0) positional embedding (name matches in many repos; keep optional)
    for name in ("decoder_pos_embed", "decoder.position_embeddings", "decoder_positional_embeddings"):
        t_name = f"{t_pref}{name}"
        if t_name in timm_sd:
            # try the most common HF name
            candidate_names = [
                f"{h_dec_pref}decoder_pos_embed",
                f"{h_dec_pref}position_embeddings",
                f"{h_dec_pref}decoder_position_embeddings",
            ]
            for cn in candidate_names:
                if cn in hf_model.state_dict() or cn.endswith("decoder_pos_embed"):
                    out[cn] = timm_sd[t_name]
                    break
            break

    # 1) embed linear
    for w_or_b in ("weight", "bias"):
        t_name = f"{t_pref}decoder_embed.{w_or_b}"
        h_name = f"{h_dec_pref}decoder_embed.{w_or_b}"
        if t_name in timm_sd and h_name in hf_model.state_dict():
            out[h_name] = timm_sd[t_name]

    # 2) per-block mappings
    # infer number of blocks from timm sd
    num_blocks = max(
        (int(k[len(t_pref) + len("decoder_blocks."):].split(".")[0])
         for k in t_keys if k.startswith(f"{t_pref}decoder_blocks.")),
        default=-1
    ) + 1

    for i in range(num_blocks):
        # norms
        for norm_src, norm_dst in (
            (f"decoder_blocks.{i}.norm1", f"decoder_layers.{i}.layernorm_before"),
            (f"decoder_blocks.{i}.norm2", f"decoder_layers.{i}.layernorm_after"),
        ):
            for w_or_b in ("weight", "bias"):
                t_name = f"{t_pref}{norm_src}.{w_or_b}"
                h_name = hk(f"{norm_dst}.{w_or_b}")
                if t_name in timm_sd and h_name in hf_model.state_dict():
                    out[h_name] = timm_sd[t_name]

        # attention qkv -> split into q, k, v
        t_qkv_w = f"{t_pref}decoder_blocks.{i}.attn.qkv.weight"
        t_qkv_b = f"{t_pref}decoder_blocks.{i}.attn.qkv.bias"
        if t_qkv_w in timm_sd:
            qkv_w = timm_sd[t_qkv_w]  # [3*D, D]
            D = qkv_w.shape[1]
            assert qkv_w.shape[0] % 3 == 0, "qkv weight first dim not divisible by 3"
            dq = qkv_w.shape[0] // 3
            q_w, k_w, v_w = qkv_w[:dq], qkv_w[dq:2*dq], qkv_w[2*dq:]
            out[hk(f"decoder_layers.{i}.attention.attention.query.weight")] = q_w
            out[hk(f"decoder_layers.{i}.attention.attention.key.weight")]   = k_w
            out[hk(f"decoder_layers.{i}.attention.attention.value.weight")] = v_w

        if t_qkv_b in timm_sd:
            qkv_b = timm_sd[t_qkv_b]  # [3*D]
            dq = qkv_b.shape[0] // 3
            q_b, k_b, v_b = qkv_b[:dq], qkv_b[dq:2*dq], qkv_b[2*dq:]
            out[hk(f"decoder_layers.{i}.attention.attention.query.bias")] = q_b
            out[hk(f"decoder_layers.{i}.attention.attention.key.bias")]   = k_b
            out[hk(f"decoder_layers.{i}.attention.attention.value.bias")] = v_b

        # attention output proj -> HF's attention.output.dense
        for w_or_b in ("weight", "bias"):
            t_name = f"{t_pref}decoder_blocks.{i}.attn.proj.{w_or_b}"
            h_name = hk(f"decoder_layers.{i}.attention.output.dense.{w_or_b}")
            if t_name in timm_sd and h_name in hf_model.state_dict():
                out[h_name] = timm_sd[t_name]

        # MLP: fc1 -> intermediate.dense, fc2 -> output.dense
        for (src, dst) in (("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense")):
            for w_or_b in ("weight", "bias"):
                t_name = f"{t_pref}decoder_blocks.{i}.{src}.{w_or_b}"
                h_name = hk(f"decoder_layers.{i}.{dst}.{w_or_b}")
                if t_name in timm_sd and h_name in hf_model.state_dict():
                    out[h_name] = timm_sd[t_name]

    # 3) tail norm
    for w_or_b in ("weight", "bias"):
        t_name = f"{t_pref}decoder_norm.{w_or_b}"
        h_name = f"{h_dec_pref}decoder_norm.{w_or_b}"
        if t_name in timm_sd and h_name in hf_model.state_dict():
            out[h_name] = timm_sd[t_name]

    # 4) predictor (to pixels/patch dim)
    for w_or_b in ("weight", "bias"):
        t_name = f"{t_pref}decoder_pred.{w_or_b}"
        h_name = f"{h_dec_pref}decoder_pred.{w_or_b}"
        if t_name in timm_sd and h_name in hf_model.state_dict():
            out[h_name] = timm_sd[t_name]

    out["mask_token"] = timm_sd.get(f"mask_token", None)

    return out



class BatchNormProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.embed_size)
        )

    def forward(self, x):
        proj_hidden = self.proj(x)
        return proj_hidden


@typechecked
class VisionTransformer(BaseLightningModel):
    """Vision Transformer implementation."""

    def __init__(self, config):
        super().__init__(config)
        # Set up ViT architecture
        vit_mae_config = ViTMAEConfig(**config['model']['model_params'])
        
        # ------------------------------------
        self.vit_mae = ViTMAE(vit_mae_config)
        hf_encoder_sd, timm_decoder_sd, config_info = convert_timm_mae_to_hf("/work/nvme/bfsx/yzhang39/mae/vit-s/checkpoint-60.pth")
        missing, unexpected = self.vit_mae.vit.load_state_dict(hf_encoder_sd, strict=False)
        assert len(missing) == 0, f"Missing keys from the model: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys in state_dict: {unexpected}"
        hf_decoder_sd = map_timm_decoder_to_hf(self.vit_mae.decoder, timm_decoder_sd)
        missing, unexpected = self.vit_mae.decoder.load_state_dict(hf_decoder_sd, strict=False)
        assert len(missing) == 0, f"Missing keys from the model: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys in state_dict: {unexpected}"
        # ------------------------------------
        
        self.mask_ratio = config['model']['model_params']['mask_ratio']
        # contrastive loss
        if config['model']['model_params']['use_infoNCE']:
            self.proj = BatchNormProjector(vit_mae_config)
            if self.config['model']['model_params']['temp_scale']:
                self.temperature = nn.Parameter(torch.ones([]) * np.log(1))

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels img_height img_width'],
    ) -> Dict[str, torch.Tensor]:
        results_dict = self.vit_mae(pixel_values=x, return_recon=True)
        if self.config['model']['model_params']['use_infoNCE']:
            cls_token = results_dict['latents'][:, 0, :]
            proj_hidden = self.proj(cls_token)
            # normalize projection
            z = proj_hidden / proj_hidden.norm(dim=-1, keepdim=True)
            results_dict['z'] = z
            results_dict['cls_token'] = cls_token

        return results_dict

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        x = batch_dict['image']
        results_dict = self.forward(x)
        if return_images:
            results_dict['images'] = x
        return results_dict

    def compute_loss(
        self,
        stage: str,
        **kwargs,
    ) -> tuple[torch.tensor, list[dict]]:
        assert 'loss' in kwargs, "Loss is not in the kwargs"
        mse_loss = kwargs['loss']
        # add all losses here for logging
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss.clone()}
        ]
        loss = mse_loss
        if self.config['model']['model_params']['use_infoNCE']:
            z = kwargs['z']
            sim_matrix = z @ z.T
            if self.config['model']['model_params']['temp_scale']:
                sim_matrix /= self.temperature.exp()
            loss_dict = batch_wise_contrastive_loss(sim_matrix)
            loss_dict['infoNCE_loss'] *= self.config['model']['model_params']['infoNCE_weight']
            log_list.append({
                'name': f'{stage}_infoNCE',
                'value': loss_dict['infoNCE_loss']
            })
            log_list.append({
                'name': f'{stage}_infoNCE_percent_correct',
                'value': loss_dict['percent_correct']
            })
            loss += loss_dict['infoNCE_loss']
        return loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        # set mask_ratio to 0 for inference
        self.vit_mae.config.mask_ratio = 0
        # get model outputs
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        # reset mask_ratio to the original value
        self.vit_mae.config.mask_ratio = self.mask_ratio
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict


class ViTMAE(ViTMAEForPreTraining):
    # Overriding the forward method to return the latent and loss
    # This is used for training and inference
    # Huggingface Transformer library
    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_latent: bool = False,
        return_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Setting default for return_dict based on the configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (self.training or self.config.mask_ratio > 0) or return_recon:
            outputs = self.vit(
                pixel_values,
                noise=noise,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            latent = outputs.last_hidden_state
        else:
            # use for fine-tuning, or inference
            # mask_ratio = 0
            embedding_output, mask, ids_restore = self.vit.embeddings(pixel_values)
            embedding_output_ = embedding_output[:, 1:, :]  # no cls token
            # unshuffle the embedding output
            index = ids_restore.unsqueeze(-1).repeat(
                1, 1, embedding_output_.shape[2]
            ).to(embedding_output_.device)
            embedding_output_ = torch.gather(embedding_output_, dim=1, index=index)
            # add cls token back
            embedding_output = torch.cat((embedding_output[:, :1, :], embedding_output_), dim=1)
            encoder_outputs = self.vit.encoder(
                embedding_output,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            latent = self.vit.layernorm(sequence_output)
            if not return_latent:
                # return the cls token and 0 loss if not return_latent
                return latent[:, 0], 0
        if return_latent:
            return latent
        # extract cls latent
        cls_latent = latent[:, 0]  # shape (batch_size, hidden_size)
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits
        # shape (batch_size, num_patches, patch_size*patch_size*num_channels)
        loss = self.forward_loss(pixel_values, logits, mask)
        if return_recon:
            return {
                'latents': latent,
                'loss': loss,
                'reconstructions': self.unpatchify(logits),
            }
        return {
            'latents': cls_latent,
            'loss': loss,
            'logits': logits,
        }


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i+1)] == labels) / len(labels)
    return topsum


def batch_wise_contrastive_loss(sim_matrix):
    N = sim_matrix.shape[0]
    # remove the diagonal from the sim_matrix
    mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix = sim_matrix[~mask].view(N, N-1)
    labels = torch.arange(N).to(sim_matrix.device)
    labels_i, labels_j = labels[:N//2], labels[N//2:] - 1
    labels = torch.cat([labels_j, labels_i]).to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    percent_correct = topk(sim_matrix, labels, k=1)
    return {
        "infoNCE_loss": loss,
        "percent_correct": percent_correct,
    }
