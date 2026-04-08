import copy

import numpy as np
import torch


def test_pca_autoencoder_from_pickle(config_ae, tmp_path):
    from beast.models.pca import GPU_PCA, PCAAutoencoder, save_pca_model

    config = copy.deepcopy(config_ae)
    config['model']['model_class'] = 'pca'
    h = w = 224
    c = 3
    d = c * h * w
    k = 16
    rng = np.random.default_rng(0)
    mean = rng.standard_normal(d).astype(np.float32)
    a = rng.standard_normal((k, d)).astype(np.float32)
    q, _ = np.linalg.qr(a.T)
    components = q.T[:k]
    ev = np.ones(k, dtype=np.float32)
    gpu_pca = GPU_PCA(components, mean, ev)
    pkl = tmp_path / 'pca.pkl'
    save_pca_model(gpu_pca, pkl)

    config['model']['model_params'] = {
        'image_size': h,
        'num_channels': c,
        'pca_pickle_path': str(pkl),
    }

    model = PCAAutoencoder(config)
    x = torch.randn(4, c, h, w)
    xhat, z = model(x)
    assert z.shape == (4, k)
    assert xhat.shape == x.shape


def test_pca_autoencoder_zero_init(config_ae):
    from beast.models.pca import PCAAutoencoder

    config = copy.deepcopy(config_ae)
    config['model']['model_class'] = 'pca'
    config['model']['model_params'] = {
        'image_size': 32,
        'num_channels': 3,
        'n_components': 5,
    }
    model = PCAAutoencoder(config)
    assert model.pca_mean.requires_grad and model.pca_components.requires_grad
    x = torch.randn(2, 3, 32, 32)
    xhat, z = model(x)
    assert z.shape == (2, 5)
    assert xhat.shape == x.shape
    loss = torch.nn.functional.mse_loss(xhat, x)
    loss.backward()
    assert model.pca_mean.grad is not None and model.pca_components.grad is not None


def test_pca_predict_step_keys(config_ae, tmp_path):
    from beast.models.pca import GPU_PCA, PCAAutoencoder, save_pca_model

    config = copy.deepcopy(config_ae)
    config['model']['model_class'] = 'pca'
    h = w = 32
    c = 3
    d = c * h * w
    k = 4
    rng = np.random.default_rng(1)
    mean = rng.standard_normal(d).astype(np.float32)
    a = rng.standard_normal((k, d)).astype(np.float32)
    q, _ = np.linalg.qr(a.T)
    components = q.T[:k]
    save_pca_model(GPU_PCA(components, mean, np.ones(k, dtype=np.float32)), tmp_path / 'p.pkl')
    config['model']['model_params'] = {
        'image_size': h,
        'num_channels': c,
        'pca_pickle_path': str(tmp_path / 'p.pkl'),
    }
    model = PCAAutoencoder(config)
    batch = {
        'image': torch.randn(1, c, h, w),
        'video': 'v0',
        'idx': 0,
        'image_path': '/tmp/x.png',
    }
    out = model.predict_step(batch, 0)
    assert 'latents' in out and 'reconstructions' in out and 'metadata' in out
