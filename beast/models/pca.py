"""Linear autoencoder in PCA form for the Lightning training / predict pipeline.

``latents = (x - μ) Wᵀ``, ``x̂ = latents W + μ`` with trainable ``μ`` and ``W`` (same
shape as PCA mean and components). Optional pickle initializes from a fitted PCA; afterward
SGD / Adam updates the subspace like a shallow linear autoencoder (not constrained to stay
orthonormal). Forward contract matches ``ResnetAutoencoder``.

Optional CUDA / CuPy helpers can fit PCA offline for initialization.
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from beast.models.base import BaseLightningModel


class GPU_PCA:
    """PCA parameters held in NumPy (compatible with :func:`save_pca_model` / :func:`load_pca_model`)."""

    def __init__(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance: np.ndarray,
    ) -> None:
        self.components_ = np.asarray(components, dtype=np.float32)
        self.mean_ = np.asarray(mean, dtype=np.float32).ravel()
        self.explained_variance_ = np.asarray(explained_variance, dtype=np.float32).ravel()
        self.n_components_ = int(self.components_.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project rows of *X* onto principal components (NumPy)."""
        X = np.asarray(X, dtype=np.float32)
        centered = X - self.mean_
        return centered @ self.components_.T


@typechecked
def save_pca_model(pca_model: GPU_PCA, output_path: str | Path) -> None:
    model_dict: dict[str, Any] = {
        'components_': np.asarray(pca_model.components_, dtype=np.float32),
        'mean_': np.asarray(pca_model.mean_, dtype=np.float32),
        'explained_variance_': np.asarray(pca_model.explained_variance_, dtype=np.float32),
        'n_components_': pca_model.n_components_,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_dict, f)


@typechecked
def load_pca_model(input_path: str | Path) -> GPU_PCA:
    with open(input_path, 'rb') as f:
        model_dict = pickle.load(f)
    return GPU_PCA(
        components=model_dict['components_'],
        mean=model_dict['mean_'],
        explained_variance=model_dict['explained_variance_'],
    )


def train_incremental_pca(
    dataloader: Any,
    n_components: int,
    batch_size: int,
    device: torch.device,
) -> GPU_PCA:
    """Fit PCA incrementally on GPU using CuPy SVD (requires ``cupy``).

    *dataloader* batches must include ``'ref'`` video tensors ``(N, C, H, W)``, as in
    contrastive setups; otherwise use :class:`PCAAutoencoder` with data saved from a
    standard ``image`` pipeline and fit sklearn offline.

    Parameters
    ----------
    dataloader:
        PyTorch dataloader yielding dict batches with key ``ref``.
    n_components:
        Number of retained components.
    batch_size:
        Unused; kept for API compatibility with older scripts.
    device:
        Torch device (only used to move tensors before NumPy/CuPy).

    Returns
    -------
    GPU_PCA
        Fitted model in NumPy.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            'train_incremental_pca requires optional dependency `cupy`. '
            'Install cupy matching your CUDA, or fit PCA with sklearn and load via pca_pickle_path.'
        ) from e

    from tqdm import tqdm

    n_samples = 0
    mean: np.ndarray | None = None
    components_gpu = None
    explained_variance_gpu = None

    total_batches = len(dataloader)
    last_2nd_batch = None

    for i, batch in enumerate(
        tqdm(dataloader, desc='Training PCA', total=total_batches),
    ):
        if i == total_batches - 2:
            last_2nd_batch = deepcopy(batch)
            continue
        if i == total_batches - 1:
            video = torch.cat([last_2nd_batch['ref'], batch['ref']], dim=0).to(device)
        else:
            video = batch['ref'].to(device)

        batch_data = video.reshape(video.shape[0], -1)
        batch_data = cp.asarray(batch_data.detach().cpu().numpy())

        if mean is None:
            mean = cp.zeros(batch_data.shape[1], dtype=cp.float64)
        mean = mean + cp.sum(batch_data, axis=0, dtype=cp.float64)
        n_samples += batch_data.shape[0]

        batch_data = batch_data - cp.mean(batch_data, axis=0)
        _u, s, vt = cp.linalg.svd(batch_data, full_matrices=False)

        if components_gpu is None:
            components_gpu = vt[:n_components]
            explained_variance_gpu = (s[:n_components] ** 2) / max(n_samples - 1, 1)
        else:
            components_gpu = cp.vstack([components_gpu, vt[:n_components]])
            explained_variance_gpu = cp.hstack([
                explained_variance_gpu,
                (s[:n_components] ** 2) / max(n_samples - 1, 1),
            ])
            _u2, s_final, vt_final = cp.linalg.svd(components_gpu, full_matrices=False)
            components_gpu = vt_final[:n_components]
            explained_variance_gpu = (s_final[:n_components] ** 2) / max(n_samples - 1, 1)

    mean_np = cp.asnumpy(mean / max(n_samples, 1)).astype(np.float32)
    components_np = cp.asnumpy(components_gpu).astype(np.float32)
    ev_np = cp.asnumpy(explained_variance_gpu).astype(np.float32)
    return GPU_PCA(components_np, mean_np, ev_np)


@typechecked
class PCAAutoencoder(BaseLightningModel):
    """Trainable linear autoencoder: ``latents = (x - μ) Wᵀ``, ``x̂ = latents W + μ``.

    ``μ`` and ``W`` are :class:`nn.Parameter` so reconstruction loss backprops and the optimizer
    from ``config['optimizer']`` updates them (same schedule as other ``BaseLightningModel``).

    **Initialization**

    - ``pca_pickle_path``: load mean / components from :func:`save_pca_model` as starting point.
    - Otherwise require ``n_components``: mean is zeros, ``W`` is small random (not degenerate).

    Model parameters (``config['model']['model_params']``):

    - ``image_size`` (default 224), ``num_channels`` (default 3)
    - ``n_components``: latent size (inferred from pickle if omitted)
    - ``pca_pickle_path``: optional initializer pickle
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        params = config['model']['model_params']
        image_size = int(params.get('image_size', 224))
        num_channels = int(params.get('num_channels', 3))
        self._flat_dim = num_channels * image_size * image_size

        pca_path = params.get('pca_pickle_path')
        if pca_path:
            pca = load_pca_model(Path(pca_path))
            mean_t = torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32))
            comp_t = torch.from_numpy(np.asarray(pca.components_, dtype=np.float32))
            if comp_t.shape[1] != self._flat_dim:
                raise ValueError(
                    f'PCA feature dim {comp_t.shape[1]} != '
                    f'num_channels * image_size ** 2 = {self._flat_dim}'
                )
        else:
            n_comp = params.get('n_components')
            if n_comp is None:
                raise ValueError(
                    'Provide model_params.pca_pickle_path or model_params.n_components.'
                )
            mean_t = torch.zeros(self._flat_dim, dtype=torch.float32)
            n_comp = int(n_comp)
            comp_t = torch.randn(n_comp, self._flat_dim, dtype=torch.float32)
            comp_t.mul_(1.0 / np.sqrt(self._flat_dim))

        self.pca_mean = nn.Parameter(mean_t)
        self.pca_components = nn.Parameter(comp_t)

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels height width'],
    ) -> tuple[
        Float[torch.Tensor, 'batch channels height width'],
        Float[torch.Tensor, 'batch n_components'],
    ]:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        centered = flat - self.pca_mean
        z = centered @ self.pca_components.T
        recon = z @ self.pca_components + self.pca_mean
        xhat = recon.reshape_as(x)
        return xhat, z

    def get_model_outputs(self, batch_dict: dict, return_images: bool = True) -> dict:
        x = batch_dict['image']
        xhat, z = self.forward(x)
        results_dict: dict = {'reconstructions': xhat, 'latents': z}
        if return_images:
            results_dict['images'] = x
        return results_dict

    def compute_loss(
        self,
        stage: str,
        images: Float[torch.Tensor, 'batch channels height width'],
        reconstructions: Float[torch.Tensor, 'batch channels height width'],
        latents: Float[torch.Tensor, 'batch n_components'],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, list[dict]]:
        mse_loss = F.mse_loss(images, reconstructions, reduction='mean')
        log_list: list[dict] = [{'name': f'{stage}_mse', 'value': mse_loss}]
        return mse_loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        results_dict = self.get_model_outputs(batch_dict, return_images=False)
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict
