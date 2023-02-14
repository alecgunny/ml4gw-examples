from typing import Callable, List, Optional, Tuple

import torch


class ShiftedPearsonLoss(torch.nn.Module):
    def __init__(self, max_shift: int, alpha: Optional[float] = None):
        super().__init__()
        self.max_shift = max_shift
        self.alpha = alpha

    def compute_corr(self, x, y):
        batch, channels, dim = x.shape
        # scale each channel's prediction to -1 to 1
        # to account for lack of amplitude info
        xmin = x.min(axis=-1, keepdims=True).values
        xmax = x.max(axis=-1, keepdims=True).values
        x = 2 * (x - xmin) / (xmax - xmin) - 1

        # window x before padding so that we
        # don't introduce any frequency artifacts
        window = torch.hann_window(dim, device=x.device)
        x = x * window

        # pad x along time dimension so that it has shape
        # batch x channels x (dim + 2 * max_shift)
        pad = (self.max_shift, self.max_shift)
        x = torch.nn.functional.pad(x, pad)

        # the following is just some magic to unroll
        # x into dim-length windows along its time axis
        # batch x channels x 1 x (dim + 2 * max_shift)
        x = x.unsqueeze(2)

        # batch x (channels * num_windows) x 1 x dim
        num_windows = 2 * self.max_shift + 1
        x = torch.nn.functional.unfold(x, (1, num_windows))

        # batch x channels x num_windows x dim
        x = x.reshape(batch, channels, num_windows, dim)

        # num_windows x batch x channels x dim
        x = x.transpose(0, 2).transpose(1, 2)

        # now compute the correlation between
        # each one of these windows of x and
        # the single window of y
        # de-mean
        x = x - x.mean(-1, keepdims=True)
        y = y - y.mean(-1, keepdims=True)

        # num_windows x batch x channels
        corr = (x * y).sum(axis=-1)
        norm = (x**2).sum(-1) * (y**2).sum(-1)

        return corr / norm**0.5

    def score_max(self, corr):
        values, _ = corr.max(axis=0)
        return (values**2).sum(axis=-1) ** 0.5

    def get_mask(self, corr):
        _, indices = corr.max(axis=0)
        mask = torch.arange(len(corr)).view(-1, 1, 1).to(indices.device)
        mask = 1 - 2 * torch.exp(-((mask - indices) ** 2) / self.alpha)
        return mask

    def get_weighted_corr(self, corr):
        mask = self.get_mask(corr)
        corr = (corr * mask).sum(axis=(0, -1)) / mask.sum(axis=(0, -1))
        return corr

    def forward(self, x, y):
        x = torch.flip(x, dims=(1,))
        corr = self.compute_corr(x, y)
        if self.alpha is None:
            scores = self.score_max(corr)
        else:
            scores = self.get_weighted_corr(corr)
        return -scores.mean()


def get_layers(
    input_dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    stride: int,
    groups: int,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    encode_layer = torch.nn.Conv1d(
        in_channels,
        out_channels * groups,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
    )

    # figure out how long our output will be
    # without any additional output padding,
    # then use that to specify the padding
    # we would need to match the input size
    pad = int(kernel_size // 2 * 2)
    encode_dim = (input_dim - kernel_size + pad) // stride + 1
    decode_dim = ((encode_dim - 1) * stride) + kernel_size - pad
    output_padding = input_dim - decode_dim
    decode_layer = torch.nn.ConvTranspose1d(
        out_channels * groups,
        in_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        output_padding=output_padding,
    )
    return encode_layer, decode_layer


class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        x = x.reshape(-1, *self.shape)
        return x


class Transpose(torch.nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        num_ifos: int,
        input_dim: int,
        layers: List[int],
        latent_dim: int,
        skip_connections: Optional[List[int]] = None,
        kernel_size: int = 7,
        stride: int = 2,
        norm_groups: int = 8,
        independent: bool = True,
        activation: Callable[None, torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_ifos = num_ifos
        self.layers = layers
        self.latent_dim = latent_dim
        self.skip_connections = skip_connections
        self.kernel_size = kernel_size
        self.norm_groups = norm_groups
        self.stride = stride
        self.independent = independent
        self.activation = activation

        self.build()

    def build(self):
        in_channels = self.num_ifos
        input_dim = self.input_dim + 0
        groups = 2 if self.independent else 1

        encoder_layers, decoder_layers = [], []
        for i, out_channels in enumerate(self.layers):
            encode_conv, decode_conv = get_layers(
                input_dim,
                in_channels,
                out_channels,
                self.kernel_size,
                int(self.kernel_size // 2),
                self.stride,
                groups,
            )
            encode_norm = torch.nn.GroupNorm(
                min(self.norm_groups, out_channels * groups),
                out_channels * groups,
            )
            encode_activation = self.activation()
            encoder_layers.extend(
                [encode_conv, encode_norm, encode_activation]
            )

            if i == 0:
                decoder_layers.insert(0, decode_conv)
            else:
                decode_norm = torch.nn.GroupNorm(
                    min(self.norm_groups, in_channels), in_channels
                )
                decode_activation = self.activation()
                block = [decode_conv, decode_norm, decode_activation]
                decoder_layers = block + decoder_layers

            in_channels = groups * out_channels
            pad = int(self.kernel_size // 2 * 2)
            input_dim -= self.kernel_size - pad
            input_dim //= self.stride
            input_dim += 1

        self.encoder_layers = torch.nn.ModuleList(encoder_layers)
        self.encode_middle = torch.nn.Sequential(
            Reshape(self.num_ifos, out_channels, input_dim),
            Transpose(1, 2),
            torch.nn.Linear(input_dim, self.latent_dim),
            torch.nn.GroupNorm(self.norm_groups, out_channels),
            self.activation(),
        )

        self.decode_middle = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, input_dim),
            torch.nn.GroupNorm(self.norm_groups, out_channels),
            self.activation(),
            Transpose(1, 2),
            Reshape(self.num_ifos * out_channels, input_dim),
        )
        self.decoder_layers = torch.nn.ModuleList(decoder_layers)

    def encode(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        conv_outputs = []
        for i, layer in enumerate(self.encoder_layers):
            X = layer(X)
            if not i % 3:
                conv_outputs.append(X)
        X = self.encode_middle(X)
        return X, conv_outputs

    def decode(
        self, X: torch.Tensor, residuals: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        X = self.decode_middle(X)
        for i, layer in enumerate(self.decoder_layers):
            X = layer(X)
            if self.skip_connections is None:
                continue
            elif residuals is None:
                raise ValueError("No residuals passed for skip connections")

            block_idx, layer_idx = divmod(i, 3)
            block_idx = len(residuals) - block_idx - 1
            if not layer_idx and block_idx in self.skip_connections:
                X += residuals[block_idx - 1]
        return X

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X, residuals = self.encode(X)
        # X = self.mlp(X)
        if self.skip_connections:
            X = self.decode(X, residuals)
        else:
            X = self.decode(X, None)
        return X
