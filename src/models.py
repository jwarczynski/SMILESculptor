from visualisations import prediction_visualization_figure, token_distribution_visualization_figure, \
    latent_space_visualization_figure, distance_correlation_figure

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torch.distributed import all_reduce, ReduceOp

import lightning as L


class LSTMWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True, return_sequence=False):
        super(LSTMWrapper, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first)
        self.return_sequence = return_sequence

    def forward(self, x):
        out, _ = self.lstm(x)
        if self.return_sequence:
            return out
        if len(out.shape) == 3:  # btached input and no return sequence
            return out[:, -1, :]
        return out[-1, :]  # no batched input and no return sequence


class BaselineVAE(nn.Module):
    def __init__(self, input_dim, seq_len, encoding_dim=512, hidden_dim=256, latent_dim=128, num_lstm_layers=3):
        super(BaselineVAE, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.SiLU(),
            LSTMWrapper(encoding_dim, encoding_dim, num_layers=num_lstm_layers, batch_first=True),
            nn.Linear(encoding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_var
        )

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            LSTMWrapper(latent_dim, latent_dim, num_layers=num_lstm_layers, batch_first=True, return_sequence=True),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, encoding_dim),
            nn.SiLU(),
            nn.Linear(encoding_dim, input_dim),
        )

    def encode(self, x, eps: float = 1e-8):
        x = self.encoder(x)
        mean, log_var = x.chunk(2, dim=-1)
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)

        dist = torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
        return dist

    def reparameterize(self, dist):
        return dist.rsample()

    def decode(self, x):
        x_reps = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        return self.decoder(x_reps)

    def forward(self, x):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        return self.decode(z), z, dist


class BaselineVAEModel(L.LightningModule):
    def __init__(self, charset_size, seq_len, encoding_dim=512, hidden_dim=256, latent_dim=128, lr=0.000251,
                 num_lstm_layers=5):
        super(BaselineVAEModel, self).__init__()
        self.save_hyperparameters()
        self.model = BaselineVAE(charset_size, seq_len, encoding_dim, hidden_dim, latent_dim,
                                 num_lstm_layers=num_lstm_layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)

        recon_loss, kl_loss, kl_weight = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * kl_weight
        self.log_dict({
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': kl_weight,
            'train_loss': loss
        }, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)

        recon_loss, kl_loss, kl_weight = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * kl_weight
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)

        recon_loss, kl_loss, kl_weight = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * kl_weight

        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def loss_function(self, x_hat, x, z, dist):
        logits = x_hat.view(-1, self.hparams.input_dim)
        targets = x.argmax(dim=-1).view(-1)
        recon_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),  # B x latent_dim
            torch.eye(z.shape[-1], device=z.device).unsqueeze(0).repeat(z.shape[0], 1, 1)  # B x latent_dim x latent_dim
        )
        # Annealing
        kl_weight = 1e-3
        kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        return recon_loss, kl_loss, kl_weight

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


class FF(nn.Module):
    def __init__(self, input_dim, growth_factor, activation=nn.SiLU):
        super(FF, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * growth_factor)),  # Ensure integer dimensions
            activation(),
        )

    def forward(self, x):
        return self.layers(x)


class FFVAE(nn.Module):
    def __init__(self, seq_len, embed_dim, charset_size, hidden_dim=2048, depth=5):
        super(FFVAE, self).__init__()
        self.seq_len = seq_len
        self.input_dim = charset_size
        self.charset_size = charset_size
        self.embedding = nn.Embedding(self.input_dim, embed_dim)
        self.ff1 = nn.Sequential(
            nn.Linear(embed_dim * seq_len, hidden_dim),
            nn.SiLU(),
        )

        # Build the encoder of given depth
        self.encoder = nn.Sequential(
            *[FF(int(hidden_dim * (1 / 2) ** i), 1 / 2) for i in range(depth)]
        )
        latent_dim = int(hidden_dim * (1 / 2) ** depth)
        self.ecnoder_last = nn.Linear(latent_dim, latent_dim * 2)

        # self.latent_dim = hidden_dim * (1/2)**depth

        self.softplus = nn.Softplus()

        self.decoder = nn.Sequential(
            *[FF(latent_dim * 2 ** i, 2) for i in range(depth)]
        )
        self.decoder_last = nn.Linear(latent_dim * 2 ** depth, charset_size * seq_len)

    def encode(self, x, eps: float = 1e-8):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.ff1(x)
        x = self.encoder(x)
        x = self.ecnoder_last(x)
        mean, log_var = x.chunk(2, dim=-1)
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)

        dist = torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
        return dist

    def decode(self, z):
        x = self.decoder(z)
        x = self.decoder_last(x)
        x = x.view(-1, self.seq_len, self.input_dim)
        assert x.shape[1] == self.seq_len
        assert x.shape[2] == self.input_dim
        return x

    def reparameterize(self, dist):
        return dist.rsample()

    def forward(self, x):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        return self.decode(z), z, dist


class LightningFFVAE(L.LightningModule):
    def __init__(self, seq_len, embed_dim, charset_size, hidden_dim=2048, depth=5, lr=1e-3):
        super(LightningFFVAE, self).__init__()
        self.save_hyperparameters()
        self.model = FFVAE(seq_len, embed_dim, charset_size, hidden_dim, depth)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)

        recon_loss, kl_loss, kl_weight = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * kl_weight
        self.log_dict({
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': kl_weight,
            'train_loss': loss
        }, prog_bar=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)

        recon_loss, kl_loss, kl_weight = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * kl_weight
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)

        recon_loss, kl_loss, kl_weight = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * kl_weight

        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def loss_function(self, x_hat, x, z, dist):
        logits = x_hat.view(-1, self.hparams.charset_size)
        targets = x.view(-1)
        recon_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')
        # recon_loss = nn.functional.cross_entropy(logits, targets, reduction='none').reshape(x.shape).sum(dim=1).mean()
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            torch.eye(z.shape[-1], device=z.device).unsqueeze(0).repeat(z.shape[0], 1, 1)
        )
        # Annealing
        kl_weight = 1e-3
        kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        return recon_loss, kl_loss, kl_weight

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


class VAEEncoder(nn.Module):
    def __init__(self, emb_dim, num_lstm_layers=8, lstm_hidden_dim=512):
        super(VAEEncoder, self).__init__()
        self.lstms = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                             batch_first=True)
        self.ff1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 4)
        self.ff2 = nn.Linear(lstm_hidden_dim // 4, lstm_hidden_dim // 16)

    def forward(self, x):
        _, (h, _) = self.lstms(x)  # h: (num_layers, batch, hidden_dim)
        h = h.transpose_(0, 1)  # h: (batch, num_layers, hidden_dim)

        h = self.ff1(h)
        h = nn.functional.silu(h)
        h = self.ff2(h)

        return h.view(x.size(0), -1)


class VAEDecoder(nn.Module):
    def __init__(self, seq_len, latent_dim, num_lstm_layers=8, lstm_hidden_dim=512):
        super(VAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True)
        self.ff1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim * 2)
        self.ff2 = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim * 2)

    def forward(self, x):
        x = x.view(x.size(0), self.num_lstm_layers, -1)
        _, (h, _) = self.lstm(x)
        h = h.transpose_(0, 1)  # h: (batch, num_layers, hidden_dim)
        h = self.ff1(h)
        h = nn.functional.silu(h)
        h = self.ff2(h)
        h = h.view(x.size(0), -1)

        return h


class YetAnotherVAE(nn.Module):
    def __init__(self, charset_size, seq_len, embed_dim, num_lstm_layers, lstm_hidden_dim):
        super(YetAnotherVAE, self).__init__()
        self.charset_size = charset_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.latent_dim = lstm_hidden_dim // 16

        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.encoder = VAEEncoder(embed_dim)
        self.decoder = VAEDecoder(seq_len, self.latent_dim, num_lstm_layers, lstm_hidden_dim)
        self.cls_head = nn.Linear(lstm_hidden_dim * 2 * num_lstm_layers, seq_len * charset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.cls_head(x)
        return x


class YetAnotherVAELightning(L.LightningModule):
    def __init__(self, charset_size, seq_len, embed_dim, num_lstm_layers, lstm_hidden_dim, lr=3e-4):
        super(YetAnotherVAELightning, self).__init__()
        self.save_hyperparameters()
        self.model = YetAnotherVAE(charset_size, seq_len, embed_dim, num_lstm_layers, lstm_hidden_dim)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.model(x)

        recon_loss = self.loss_function(x_hat, x)
        self.log('train_loss', recon_loss, prog_bar=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.model(x)

        recon_loss = self.loss_function(x_hat, x)
        self.log('val_loss', recon_loss, prog_bar=True)
        return recon_loss

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.model(x)

        recon_loss = self.loss_function(x_hat, x)
        self.log('test_loss', recon_loss, prog_bar=True)
        return recon_loss

    def loss_function(self, x_hat, x):
        logits = x_hat.view(-1, self.hparams.charset_size)
        targets = x.view(-1)
        recon_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')
        return recon_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


class LSTMVAEEncoder(nn.Module):
    def __init__(self, emb_dim, num_lstm_layers=8, lstm_hidden_dim=512):
        super(LSTMVAEEncoder, self).__init__()
        self.lstms = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                             batch_first=True)
        self.ff1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 4)
        self.ff2 = nn.Linear(lstm_hidden_dim // 4, lstm_hidden_dim // 16 * 2)  # 2 for mean and log_var

        self.softplus = nn.Softplus()

    def forward(self, x, eps: float = 1e-8):
        _, (h, _) = self.lstms(x)  # h: (num_layers, batch, hidden_dim)
        h = h.transpose_(0, 1)  # h: (batch, num_layers, hidden_dim)

        h = self.ff1(h)
        h = nn.functional.silu(h)
        h = self.ff2(h)

        h = h.view(x.size(0), -1)

        mean, log_var = h[:, ::2], h[:, 1::2]
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)

        dist = torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)
        return dist


class LSTMVAEDecoder(nn.Module):
    def __init__(self, charset_size, seq_len, latent_dim, num_lstm_layers=8, lstm_hidden_dim=512):
        super(LSTMVAEDecoder, self).__init__()
        self.seq_len = seq_len
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True)
        self.ff1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim * 2)
        self.ff2 = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim * 2)
        self.cls_head = nn.Linear(lstm_hidden_dim * 2 * num_lstm_layers, seq_len * charset_size)

    def forward(self, x):
        x = x.view(x.size(0), self.num_lstm_layers, -1)
        _, (h, _) = self.lstm(x)
        h = h.transpose_(0, 1)  # h: (batch, num_layers, hidden_dim)
        h = self.ff1(h)
        h = nn.functional.silu(h)
        h = self.ff2(h)
        h = h.view(x.size(0), -1)

        return self.cls_head(h)


# noinspection SpellCheckingInspection
class LSTMVAE(nn.Module):
    def __init__(self, charset_size, seq_len, embed_dim, num_lstm_layers, lstm_hidden_dim):
        super(LSTMVAE, self).__init__()

        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.encoder = LSTMVAEEncoder(embed_dim, num_lstm_layers, lstm_hidden_dim)
        self.decoder = LSTMVAEDecoder(charset_size, seq_len, lstm_hidden_dim // 16, num_lstm_layers, lstm_hidden_dim)

    def reparameterize(self, dist):
        return dist.rsample()

    def forward(self, x):
        x = self.embedding(x)
        dist = self.encoder(x)
        z = self.reparameterize(dist)
        return self.decoder(z), z, dist


class LSTMVAELightning(L.LightningModule):
    def __init__(self, charset_size, seq_len, embed_dim, num_lstm_layers, lstm_hidden_dim, lr=3e-4, kl_weight=1e-3):
        super(LSTMVAELightning, self).__init__()
        self.save_hyperparameters()
        self.model = LSTMVAE(charset_size, seq_len, embed_dim, num_lstm_layers, lstm_hidden_dim)

    def forward(self, x):
        return self.model(x)

    def forward_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, z, dist = self.model(x)
        recon_loss, kl_loss = self.loss_function(x_hat, x, z, dist)
        loss = recon_loss + kl_loss * self.hparams.kl_weight

        return loss, recon_loss, kl_loss, self.hparams.kl_weight

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, kl_weight = self.forward_step(batch, batch_idx)
        self.log_dict({
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': kl_weight,
            'train_loss': loss
        })

        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, kl_weight = self.forward_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, kl_weight = self.forward_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def loss_function(self, x_hat, x, z, dist):
        logits = x_hat.view(-1, self.hparams.charset_size)
        targets = x.view(-1)
        recon_loss = nn.functional.cross_entropy(logits, targets, reduction='mean')

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            torch.eye(z.shape[-1], device=z.device).unsqueeze(0).repeat(z.shape[0], 1, 1)
        )
        kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        return recon_loss, kl_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


def calculate_conv_output_size(input_size, conv_params):
    """
    Dynamically calculates the output size after a sequence of Conv1d layers.
    """
    for layer_params in conv_params.values():
        kernel_size = layer_params.get("kernel_size", 3)
        stride = layer_params.get("stride", 1)
        padding = layer_params.get("padding", 0)

        # PyTorch formula for Conv1D output size
        input_size = (input_size - kernel_size + 2 * padding) // stride + 1

    return input_size


class MOAVEncoder(nn.Module):
    def __init__(self, params, charset_length, max_length):
        super(MOAVEncoder, self).__init__()
        self.charset_length = charset_length
        self.max_length = max_length
        self.softplus = nn.Softplus()

        # Build convolutional layers
        conv_layers = []
        in_channels = charset_length  # Start with the input size as the number of input channels
        for idx, (layer_name, layer_params) in enumerate(params["conv_layers"].items()):
            out_channels = layer_params["out_channels"]
            kernel_size = layer_params.get("kernel_size", 9)
            stride = layer_params.get("stride", 1)
            padding = layer_params.get("padding", 0)
            activation = getattr(nn, layer_params["activation"])

            # Add Conv1d layer
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding))

            if layer_params["batch_norm"]:
                conv_layers.append(nn.BatchNorm1d(out_channels))

            conv_layers.append(activation())
            in_channels = out_channels  # Update the input channels for the next layer

        self.conv_layers = nn.Sequential(*conv_layers)

        conv_output_size = calculate_conv_output_size(self.max_length, params["conv_layers"]) * in_channels
        #reshape in forward call

        # Dense Layers
        dense_layers = []
        in_features = conv_output_size
        for layer_name, layer_params in params["dense_layers"].items():
            out_features = layer_params["dimension"]
            dense_layers.append(nn.Linear(in_features, out_features))

            if "dropout" in layer_params:
                dense_layers.append(nn.Dropout(layer_params["dropout"]))
            if "batch_norm" in layer_params:
                dense_layers.append(nn.BatchNorm1d(out_features))

            dense_layers.append(getattr(nn, layer_params["activation"])())
            in_features = out_features

        self.dense_layers = nn.Sequential(*dense_layers)

        latent_dim = params["latent_dimension"]
        if "sampling_layers" in params:
            params = params["sampling_layers"]
            activation = getattr(nn, params["activation"])
            self.sampling_layers = nn.Sequential(
                nn.Linear(in_features, latent_dim * 2),
                activation()
            )

    def forward(self, x, eps=1e-2):
        x = self.conv_layers(x.transpose(-1, -2))
        x = x.transpose(-1, -2)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense_layers(x)
        z_mean, z_log_var = self.sampling_layers(x).chunk(2, dim=-1)

        # Compute the standard deviation (scale) directly using exp(log_var / 2)
        std = torch.exp(z_log_var / 2)

        # Sample epsilon from a standard normal distribution
        epsilon = torch.randn_like(std)

        # Apply the reparameterization trick
        z = z_mean + std * epsilon

        return z_mean, z_log_var, z


class GruWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first=True):
        super(GruWrapper, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=batch_first)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class MOVAEDecoder(nn.Module):
    def __init__(self, params, charset_length, max_length):
        super(MOVAEDecoder, self).__init__()

        self.charset_length = charset_length
        self.max_length = max_length

        dense_layers = []
        input_dim = params["latent_dimension"]
        for _, layer_params in params["dense_layers"].items():
            dense_layers.append(nn.Linear(input_dim, layer_params["dimension"]))
            if "dropout" in layer_params:
                dense_layers.append(nn.Dropout(layer_params["dropout"]))
            if "batch_norm" in layer_params:
                dense_layers.append(nn.BatchNorm1d(layer_params["dimension"]))
            dense_layers.append(getattr(nn, layer_params["activation"])())
            input_dim = layer_params["dimension"]
        self.dense_layers = nn.Sequential(*dense_layers)

        recurrent_layers = []
        recurrent_layers_params = params["recurrent_layers"]
        num_recurrent_layers = recurrent_layers_params["num_layers"]
        h_dim = recurrent_layers_params["dimension"]
        for i in range(num_recurrent_layers):
            recurrent_layers.append(GruWrapper(input_dim, h_dim, 1, batch_first=True))
            input_dim = h_dim   # Update the input dimension for the next layer
        self.recurrent_layers = nn.Sequential(*recurrent_layers)

        self.output_layer = nn.Linear(h_dim, self.charset_length)

    def forward(self, x):
        x = self.dense_layers(x)
        x = x.unsqueeze(1).repeat(1, self.max_length, 1)
        x = self.recurrent_layers(x)  # batch, seq_len, h_dim
        batch_size, seq_len, h_dim = x.size()
        x = x.contiguous().view(batch_size * seq_len, h_dim)
        x = self.output_layer(x)
        x = x.view(batch_size, seq_len, self.charset_length)
        return x


class MOVAE(nn.Module):
    def __init__(self, encoder_params, decoder_params, charset_length, max_length):
        super(MOVAE, self).__init__()

        self.charset_length = charset_length
        self.max_length = max_length

        self.encoder = MOAVEncoder(encoder_params, charset_length, max_length)
        self.decoder = MOVAEDecoder(decoder_params, charset_length, max_length)

    def reparameterize(self, dist):
        return dist.rsample()

    def forward(self, x, eps=1e-2):
        z_mean, z_log_var, z = self.encoder(x)
        # z = self.reparameterize(dist)
        return self.decoder(z), z, z_mean, z_log_var


class MOVVAELightning(L.LightningModule):
    def __init__(self, params, charset_size, seq_len, lr=1e-3, kl_weight=1e-3, int_to_char=None):
        super(MOVVAELightning, self).__init__()
        self.save_hyperparameters()
        self.model = MOVAE(params["encoder_params"], params["decoder_params"], charset_size, seq_len)
        self.training_params = {
            "rlr_patience": 3,
            "rlr_factor": 0.5,
            "rlr_initial_lr": lr,
            "rlr_mindelta": 1e-7,
        }

        # Store the character mapping for visualization
        self.int_to_char = int_to_char or {}

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, x_hat, x):
        """
        Compute metrics: percentage of perfectly reconstructed sequences
        and percentage of correctly recognized elements.
        """
        predictions = x_hat.argmax(-1)  # Take argmax of predictions
        targets = x.argmax(-1)  # Convert one-hot encoded targets to indices

        # Perfectly reconstructed molecules
        is_perfect = (predictions == targets).all(dim=1)
        perfect_recon_percentage = is_perfect.float().mean() * 100  # Percentage

        # Correctly recognized elements
        correct_elements_percentage = (predictions == targets).float().mean() * 100  # Percentage

        return perfect_recon_percentage, correct_elements_percentage

    def get_current_lr(self):
        """
        Retrieve the current learning rate from the optimizer.
        """
        return self.optimizers().param_groups[0]["lr"]

    def log_metrics(self, batch_idx, loss, recon_loss, kl_loss, metrics, prefix, lr=None, z_mean=None, z_log_var=None, x_hat=None,
                    x=None):
        """
        Enhanced logging method to include z_mean and z_log_var statistics
        and optional prediction visualizations
        """
        perfect_recon, correct_elements = metrics
        log_dict = {
            f"{prefix}/loss": loss,
            f"{prefix}/recon_loss": recon_loss,
            f"{prefix}/kl_loss": kl_loss,
            f"{prefix}/perfect_recon": perfect_recon,
            f"{prefix}/correct_elements": correct_elements
        }

        # Log z_mean and z_log_var statistics if provided
        if z_mean is not None and z_log_var is not None:
            log_dict[f"{prefix}/z_mean_norm"] = torch.norm(z_mean)
            log_dict[f"{prefix}/z_log_var_mean"] = z_log_var.mean()
            log_dict[f"{prefix}/z_log_var_std"] = z_log_var.std()

            # Optional: Visualize z distribution
            self.log_latent_space_visualization(z_mean, prefix)

        if lr is not None:
            log_dict[f"{prefix}/lr"] = lr

        # Log metrics
        if prefix == "train":
            self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log_dict(log_dict, on_epoch=True, prog_bar=True)

        # Visualization of predictions
        if x_hat is not None and x is not None and (prefix != "train" or batch_idx % 100 == 0):
            self.log_prediction_visualization(x_hat, x, prefix)
            self.log_token_distribution_visualization(x_hat, x, prefix)
            # self.log_distance_correlation_visualization(z_mean, x, prefix)

    def log_distance_correlation_visualization(self, z_mean, smiles_vectors, prefix):
        """
        Generate and log multiple visualizations of the latent space

        Args:
        z_mean (torch.Tensor): Latent space mean representations
        smiles_vectors (np.ndarray): One-hot encoded SMILES vectors
        prefix (str): Logging prefix for tracking
        """

        img = distance_correlation_figure(z_mean, smiles_vectors, max_samples=16)
        self.logger.log_image(f'{prefix}/distance_correlation', images=[img])

    def log_token_distribution_visualization(self, x_hat, x, prefix):
        img = token_distribution_visualization_figure(x_hat, x, self.int_to_char)
        self.logger.log_image(f'{prefix}/token_distribution', images=[img])

    def log_latent_space_visualization(self, z_mean, prefix):
        """
        Generate and log visualization of the latent space mean representations
        """
        img = latent_space_visualization_figure(z_mean)
        self.logger.log_image(f'{prefix}/latent_space', images=[img])

    def log_prediction_visualization(self, x_hat, x, prefix):
        """
        Generate and log visualization of the model's predictions
        :param x_hat: model predictions logits
        :param x: one hot target sequences
        :param prefix: logging prefix
        :return: None
        """
        img = prediction_visualization_figure(x_hat, x, self.int_to_char)
        self.logger.log_image(f'{prefix}/predictions', images=[img])

    def forward_pass(self, batch):
        """
        Modified forward pass to return additional information
        """
        x = batch[0]
        x_hat, z, z_mean, z_log_var = self.model(x)
        recon_loss, kl_loss = self.loss_function(x_hat, x, z_mean, z_log_var)
        loss = recon_loss + kl_loss * self.hparams.kl_weight
        return loss, recon_loss, kl_loss, x_hat, x, z_mean, z_log_var

    def on_before_optimizer_step(self, optimizer):
        pass
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, x_hat, x, z_mean, z_log_var = self.forward_pass(batch)
        metrics = self.compute_metrics(x_hat, x)
        lr = self.get_current_lr()
        self.log_metrics(
            batch_idx,
            loss, recon_loss, kl_loss, metrics, "train",
            lr=lr, z_mean=z_mean, z_log_var=z_log_var,
            x_hat=x_hat, x=x
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, x_hat, x, z_mean, z_log_var = self.forward_pass(batch)
        metrics = self.compute_metrics(x_hat, x)
        self.log_metrics(
            batch_idx,
            loss, recon_loss, kl_loss, metrics, "val",
            z_mean=z_mean, z_log_var=z_log_var,
            x_hat=x_hat, x=x
        )
        return loss

    # Similar modification needed for test_step
    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_loss, x_hat, x, z_mean, z_log_var = self.forward_pass(batch)
        metrics = self.compute_metrics(x_hat, x)
        self.log_metrics(
            batch_idx,
            loss, recon_loss, kl_loss, metrics, "test",
            z_mean=z_mean, z_log_var=z_log_var,
            x_hat=x_hat, x=x
        )
        return loss

    def loss_function(self, x_hat, x, z_mean, z_log_var):
        # If x is one-hot, use binary cross-entropy
        recon_loss = nn.functional.binary_cross_entropy_with_logits(
            x_hat.view(-1, self.hparams.charset_size),
            x.view(-1, self.hparams.charset_size),
            reduction='mean'
        )

        # KL divergence calculation closer to TensorFlow implementation
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return recon_loss, kl_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_params["rlr_initial_lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.training_params["rlr_factor"],
            patience=self.training_params["rlr_patience"],
            min_lr=self.training_params["rlr_mindelta"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_recon_loss"  # The metric to monitor
            }
        }


class MoleculeAutoEncoder(nn.Module):
    def __init__(self, charset_length, max_length):
        super(MoleculeAutoEncoder, self).__init__()

        # Encoder
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=charset_length, out_channels=9, kernel_size=9)
        self.bn1 = nn.BatchNorm1d(9)

        self.conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.bn2 = nn.BatchNorm1d(9)

        self.conv3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)
        self.bn3 = nn.BatchNorm1d(10)

        # Flatten
        self.flatten = nn.Flatten()

        # Dense Layers
        self.dense1 = nn.Linear(10 * (max_length - 26), 436)
        self.bn_dense1 = nn.BatchNorm1d(436)
        self.dropout1 = nn.Dropout(0.083)

        # Decoder Layers
        self.decode_dense1 = nn.Linear(436, 436)
        self.decode_bn1 = nn.BatchNorm1d(436)
        self.decode_dropout1 = nn.Dropout(0.1)

        # Repeat Vector (similar to TensorFlow's RepeatVector)
        self.max_length = max_length

        # Recurrent Layer
        self.gru_hidden_size = 488
        self.gru = nn.GRU(input_size=436, hidden_size=self.gru_hidden_size, num_layers=3, batch_first=True)

        # Final layer to reconstruct one-hot encoded sequence
        self.reconstruct = nn.Linear(self.gru_hidden_size, charset_length)

    def encode(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.tanh(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.tanh(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.tanh(x)

        # Flatten
        x = self.flatten(x)

        # Dense Layers
        x = self.dense1(x)
        x = self.bn_dense1(x)
        x = torch.tanh(x)
        x = self.dropout1(x)

        return x

    def decode(self, x):
        # Decoder Dense Layers
        x = self.decode_dense1(x)
        x = self.decode_bn1(x)
        x = torch.tanh(x)
        x = self.decode_dropout1(x)

        # Repeat Vector (similar to TensorFlow's RepeatVector)
        # Unsqueeze to add sequence dimension and repeat
        x = x.unsqueeze(1).repeat(1, self.max_length, 1)

        # GRU Layers
        x, _ = self.gru(x)

        # Reconstruct one-hot encoded sequence
        x = self.reconstruct(x)

        return x

    def forward(self, x):
        # Encode
        latent = self.encode(x)

        # Decode
        reconstructed = self.decode(latent)

        return reconstructed


class MoleculeAutoEncoderLightning(L.LightningModule):
    def __init__(self,
                 charset_size,
                 seq_len,
                 int_to_char,
                 lr=1e-3):
        super().__init__()

        # Model Architecture
        self.charset_length = charset_size
        self.max_length = seq_len

        self.idx_to_char = int_to_char
        self.char_to_int = {v: k for k, v in int_to_char.items()}

        # Model
        self.model = MoleculeAutoEncoder(charset_size, seq_len)

        # Loss Function
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.char_to_int["?"])

        self.scheduler_params = {
            "rlr_patience": 3,
            "rlr_factor": 0.5,
            "rlr_initial_lr": lr,
            "rlr_mindelta": 1e-7,
        }

        self.learning_rate = lr

        # Tracking metrics
        # self.total_loss_tracker = torchmetrics.MeanMetric()
        # self.reconstruction_loss_tracker = torchmetrics.MeanMetric()

        # Classification metrics
        self.binarized_accuracy = torchmetrics.Accuracy(task='binary')
        self.binarized_precision = torchmetrics.Precision(task='binary')
        self.binarized_recall = torchmetrics.Recall(task='binary')
        self.binarized_f1 = torchmetrics.F1Score(task='binary')

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=charset_size)
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=charset_size)
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=charset_size)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=charset_size)

    def get_current_lr(self):
        """
        Retrieve the current learning rate from the optimizer.
        """
        return self.lr_schedulers().get_last_lr()[0]

    def forward(self, x):
        return self.model(x)

    def forward_step(self, batch, batch_idx, prefix):
        x = batch[0]
        y = x.clone()
        x = x.transpose(-2, -1)

        # Reconstruct
        decoded = self(x)

        # Reconstruction Loss (Binary Cross Entropy)
        bce_recon_loss = self.binary_cross_entropy(decoded, y)
        ce_recon_loss = self.cross_entropy(decoded.view(-1, self.charset_length), y.argmax(-1).view(-1))

        self.compute_and_log_metrics(bce_recon_loss, ce_recon_loss, decoded, y, batch_idx, prefix)
        return ce_recon_loss

    def compute_and_log_metrics(self, bce_recon_loss, ce_loss, decoded, y, batch_idx, prefix):
        """
        Compute and log metrics for the current step.

        Args:
            bce_recon_loss: The binary cross entropy reconstruction loss for the current step.
            ce_loss: The cross-entropy loss for the current step.
            decoded: The model's predictions.
            y: The ground truth sequences.
            batch_idx: The index of the current batch.
            prefix: The prefix for logging (e.g., 'train', 'val', 'test').

        Returns:
            recon_loss: The reconstruction loss for the current step.
        """

        # Log reconstruction loss
        self.log(f'{prefix}/binary_ce_recon_loss', bce_recon_loss, prog_bar=True, sync_dist=True)
        self.log(f'{prefix}/cross_entropy_recon_loss', ce_loss, prog_bar=True, sync_dist=True)

        # Log learning rate only if not in test phase
        if self.trainer.state.stage != "test":
            self.log(f"{prefix}/lr", self.get_current_lr(), prog_bar=True)

        # Compute metrics
        y_hat = decoded.argmax(dim=-1)
        y_classes = y.argmax(dim=-1)

        self.binarized_accuracy.update(decoded, y)
        self.binarized_precision.update(decoded, y)
        self.binarized_recall.update(decoded, y)
        self.binarized_f1.update(decoded, y)
        self.accuracy.update(y_hat, y_classes)
        self.precision.update(y_hat, y_classes)
        self.recall.update(y_hat, y_classes)
        self.f1.update(y_hat, y_classes)

        # Log predictions visualization periodically
        if batch_idx % 10 == 0:
            predictions_img = prediction_visualization_figure(decoded, y, self.charset_length, self.idx_to_char)
            token_dist_img = token_distribution_visualization_figure(decoded, y, self.idx_to_char)
            self.logger.log_image(f'{prefix}/predictions', images=[predictions_img])
            self.logger.log_image(f'{prefix}/token_distribution', images=[token_dist_img])

    def training_step(self, batch, batch_idx):
        # Additional training-specific logic (if any) can be added here
        return self.forward_step(batch, batch_idx, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, prefix='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.scheduler_params["rlr_initial_lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_params["rlr_factor"],
            patience=self.scheduler_params["rlr_patience"],
            min_lr=self.scheduler_params["rlr_mindelta"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/recon_loss"  # The metric to monitor
            }
        }

    def sync_metric(self, metric):
        """
        Synchronizes a metric across distributed processes.

        Args:
            metric (float or torch.Tensor): The metric value to synchronize.

        Returns:
            float: The synchronized metric value.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Ensure the metric is a tensor and move it to the correct device
            if not isinstance(metric, torch.Tensor):
                metric = torch.tensor(metric, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                metric = metric.clone().detach()

            # Perform distributed all-reduce operation
            all_reduce(metric, op=ReduceOp.SUM)
            metric /= torch.distributed.get_world_size()
            return metric.item()
        return metric

    def log_and_reset_metrics(self, prefix):
        """
        Logs and resets metrics with a given prefix.

        Args:
            prefix (str): The prefix for the metric names (e.g., 'epoch', 'test', 'val').
        """
        for name, metric in self.get_metrics().items():
            # Compute metric and sync if in distributed mode
            metric_value = metric.compute()
            metric_value = self.sync_metric(metric_value)

            # Log the synchronized metric
            self.logger.experiment.log({f'{prefix}/{name}': metric_value, 'epoch': self.current_epoch})

            # Reset the metric
            metric.reset()

    def get_metrics(self):
        """
        Returns a dictionary of common metrics to log and reset.
        """
        return {
            'binarized_accuracy': self.binarized_accuracy,
            'binarized_precision': self.binarized_precision,
            'binarized_recall': self.binarized_recall,
            'binarized_f1': self.binarized_f1,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }

    def on_validation_start(self) -> None:
        # Log and reset training epoch metrics
        self.log_and_reset_metrics("train_epoch")  # validation perform before train epoch_end

    def on_test_end(self):
        # Log and reset testing metrics
        self.log_and_reset_metrics('test')

    def on_validation_end(self):
        # Log and reset validation metrics
        self.log_and_reset_metrics('val')


class SimpleAutoencoderModel(L.LightningModule):
    def __init__(self, int_to_char, seq_len, charset_size, lr=1e-3, loss="ce"):
        super(SimpleAutoencoderModel, self).__init__()
        self.int_to_char = int_to_char
        self.char_to_int = {v: k for k, v in int_to_char.items()}
        self.seq_len = seq_len
        self.charset_size = charset_size
        # Encoder
        self.encoder_conv1 = nn.Conv1d(in_channels=seq_len, out_channels=seq_len - 9, kernel_size=9)
        self.bn1 = nn.BatchNorm1d(seq_len - 9)
        self.encoder_conv2 = nn.Conv1d(in_channels=seq_len - 9, out_channels=seq_len - 19, kernel_size=9)
        self.bn2 = nn.BatchNorm1d(seq_len - 19)
        self.encoder_conv3 = nn.Conv1d(in_channels=seq_len - 19, out_channels=seq_len - 26, kernel_size=9)
        self.bn3 = nn.BatchNorm1d(seq_len - 26)
        self.encoder_fc = nn.Linear((seq_len - 26) * (charset_size - 24), 196)

        # Decoder
        self.decoder_fc = nn.Linear(196, 196)
        self.bn4 = nn.BatchNorm1d(196)
        self.decoder_gru = nn.GRU(
            input_size=196, hidden_size=488, num_layers=3, batch_first=True
        )
        self.decoder_output1 = nn.Linear(
            488, self.charset_size
        )  # Output layer to match input dimensions
        self.dropout = nn.Dropout(0.2)

        # Loss
        self.loss = loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.char_to_int['?'])

    def encode(self, x):
        x = F.relu(self.bn1(self.encoder_conv1(x)))
        x = F.relu(self.bn2(self.encoder_conv2(x)))
        x = F.relu(self.bn3(self.encoder_conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        return x

    def decode(self, z):
        predictions = []
        hn = torch.zeros(3, z.size(0), 488).to(z.device)
        for i in range(self.seq_len):
            p, hn = self.decoder_gru(
                self.dropout(self.bn4(self.decoder_fc(z))).unsqueeze(1), hn
            )
            p = self.decoder_output1(p)
            predictions.append(p)

        return torch.cat(predictions, dim=1)

    def calculate_metrics(self, x, y_hat):
        mask = torch.Tensor(
            x.argmax(axis=-1) != self.char_to_int['?']
        ).unsqueeze(-1)
        ce_loss = self.ce_loss(
            y_hat.view(-1, self.charset_size),
            x.argmax(axis=-1).view(-1)
        )
        bce_loss = self.bce_loss(
            (y_hat * mask).unsqueeze(-1),
            (x * mask).unsqueeze(-1),
        )
        acc = (
            torch.Tensor((x * mask)).argmax(axis=-1).view(-1).cpu()
            == (y_hat * mask).argmax(axis=-1).view(-1).cpu()
        )

        return ce_loss, bce_loss, acc

    def training_step(self, batch, batch_nb):  # REQUIRED
        x = batch[0]
        y_hat = self.forward(x)

        ce_loss, bce_loss, acc = self.calculate_metrics(x, y_hat)

        self.log("train_ce_loss", ce_loss, sync_dist=True)
        self.log("train_bce_loss", bce_loss, sync_dist=True)
        self.log("accuracy_train", len(acc[acc == True]) / len(acc), sync_dist=True)

        if batch_nb % 10 == 0:
            predictions_img = prediction_visualization_figure(y_hat, x, len(self.int_to_char), self.int_to_char)
            token_dist_img = token_distribution_visualization_figure(y_hat, x, self.int_to_char)
            self.logger.log_image(f'train/predictions', images=[predictions_img])
            self.logger.log_image(f'train/token_distribution', images=[token_dist_img])

        if self.loss == "ce":
            return ce_loss
        elif self.loss == "bce":
            return bce_loss
        else:
            raise ValueError("Invalid loss function")

    def validation_step(self, batch, batch_nb):  # OPTIONAL
        x = batch[0]
        y_hat = self.forward(x)

        ce_loss, bce_loss, acc = self.calculate_metrics(x, y_hat)

        self.log("accuracy_val", len(acc[acc == True]) / len(acc), sync_dist=True)
        self.log("val_ce_loss", ce_loss, sync_dist=True)
        self.log("val_bce_loss", bce_loss, sync_dist=True)
        if self.loss == "ce":
            return ce_loss
        elif self.loss == "bce":
            return bce_loss
        else:
            raise ValueError("Invalid loss function")

    def test_step(self, batch, batch_nb):  # OPTIONAL
        x = batch[0]
        y_hat = self.forward(x)

        ce_loss, bce_loss, acc = self.calculate_metrics(x, y_hat)
        self.log("accuracy_test", len(acc[acc == True]) / len(acc), sync_dist=True)
        self.log("test_ce_loss", ce_loss, sync_dist=True)
        self.log("test_bce_loss", bce_loss, sync_dist=True)

        if self.loss == "ce":
            return ce_loss
        elif self.loss == "bce":
            return bce_loss
        else:
            raise ValueError("Invalid loss function")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)

        return x_recon
