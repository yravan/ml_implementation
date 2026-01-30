# Autoencoder implementations from scratch
# Covers various VAE classes for interview prep

from .autoencoder import (
    Autoencoder,
    Encoder,
    Decoder,
    mse_loss,
    bce_loss,
)

from .vae import (
    VAE,
    VAEEncoder,
    VAEDecoder,
    kl_divergence_gaussian,
    reparameterize,
)

from .beta_vae import (
    BetaVAE,
    AnnealedBetaVAE,
)

from .cvae import (
    CVAE,
    CVAEEncoder,
    CVAEDecoder,
    one_hot_encode,
)

from .vqvae import (
    VQVAE,
    VQVAEEncoder,
    VQVAEDecoder,
    VectorQuantizer,
    find_nearest_codebook,
    vq_loss,
)
