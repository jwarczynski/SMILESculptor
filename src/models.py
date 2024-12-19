from visualisations import prediction_visualization_figure, token_distribution_visualization_figure, \
    latent_space_visualization_figure, distance_correlation_figure

import torch
import torchmetrics
from torch import nn
from torch.distributed import all_reduce, ReduceOp

import lightning as L


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
                nn.Linear(in_features, latent_dim * 2),  # mean and log_var
                activation()
            )

    def forward(self, x, eps=1e-8):
        """
        Forward pass of the encoder

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_length, charset_length)
            eps (float): Small value to avoid numerical instability
        """
        x = self.conv_layers(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense_layers(x)
        z_mean, z_log_var = self.sampling_layers(x).chunk(2, dim=-1)

        std_dev = self.softplus(z_log_var) + eps
        std_dev_tril = torch.diag_embed(std_dev)

        dist = torch.distributions.MultivariateNormal(z_mean, scale_tril=std_dev_tril)
        z = dist.rsample()

        return z, z_mean, std_dev


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

        recurrent_layers_params = params["recurrent_layers"]
        num_recurrent_layers = recurrent_layers_params["num_layers"]
        h_dim = recurrent_layers_params["dimension"]
        self.recurrent_layers = nn.GRU(input_dim, h_dim, num_recurrent_layers, batch_first=True)

        self.output_layer = nn.Linear(h_dim, self.charset_length)

    def forward(self, x):
        x = self.dense_layers(x)
        x = x.unsqueeze(1).repeat(1, self.max_length, 1)
        x, _ = self.recurrent_layers(x)  # batch, seq_len, h_dim
        return self.output_layer(x)


class MOVAE(nn.Module):
    def __init__(self, encoder_params, decoder_params, charset_length, max_length):
        super(MOVAE, self).__init__()

        self.charset_length = charset_length
        self.max_length = max_length

        self.encoder = MOAVEncoder(encoder_params, charset_length, max_length)
        self.decoder = MOVAEDecoder(decoder_params, charset_length, max_length)

    def forward(self, x):
        z, z_mean, std_dev = self.encoder(x)
        # z = self.reparameterize(dist)
        return self.decoder(z), z, z_mean, std_dev


class MOVVAELightning(L.LightningModule):
    def __init__(self, params, charset_size, seq_len, lr=1e-3, kl_weight=1, int_to_char=None):
        super(MOVVAELightning, self).__init__()
        self.save_hyperparameters()
        self.model = MOVAE(params["encoder_params"], params["decoder_params"], charset_size, seq_len)

        self.training_params = {
            "rlr_patience": 3,
            "rlr_factor": 0.5,
            "rlr_initial_lr": lr,
            "rlr_mindelta": 1e-7,
        }

        self.kl_weight = kl_weight
        self.loss = nn.CrossEntropyLoss()
        self.int_to_char = int_to_char or {}
        self.charset_size = charset_size

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

    def forward_pass(self, batch):
        """
        Modified forward pass to return additional information
        """
        x = batch[0]
        x_hat, z, z_mean, std_dev = self.model(x.transpose(-1, -2))
        recon_loss, kl_loss = self.loss_function(x_hat, x, z_mean, std_dev)
        loss = recon_loss + kl_loss * self.kl_weight
        return loss, recon_loss, kl_loss, x_hat, x, z_mean, std_dev

    def loss_function(self, x_hat, x, z_mean, std_dev_hat):
        dist_std = torch.distributions.MultivariateNormal(
            torch.zeros_like(z_mean, device=z_mean.device),
            scale_tril=torch.eye(z_mean.shape[-1], device=z_mean.device).unsqueeze(0).repeat(z_mean.shape[0], 1, 1)
        )
        dist_hat = torch.distributions.MultivariateNormal(z_mean, scale_tril=torch.diag_embed(std_dev_hat))

        kl_loss = torch.distributions.kl.kl_divergence(dist_hat, dist_std).mean()
        recon_loss = self.loss(x_hat.view(-1, self.hparams.charset_size), x.argmax(-1).view(-1))

        return recon_loss, kl_loss

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

    def log_metrics(self, batch_idx, loss, recon_loss, kl_loss, metrics, prefix, lr=None, z_mean=None, z_log_var=None,
                    x_hat=None,
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

    def log_prediction_visualization(self, x_hat, x, prefix):
        """
        Generate and log visualization of the model's predictions
        :param x_hat: model predictions logits
        :param x: one hot target sequences
        :param prefix: logging prefix
        :return: None
        """
        img = prediction_visualization_figure(x_hat, x, self.charset_size, self.int_to_char)
        self.logger.log_image(f'{prefix}/predictions', images=[img])

    def log_token_distribution_visualization(self, x_hat, x, prefix):
        img = token_distribution_visualization_figure(x_hat, x, self.int_to_char)
        self.logger.log_image(f'{prefix}/token_distribution', images=[img])

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

    def log_latent_space_visualization(self, z_mean, prefix):
        """
        Generate and log visualization of the latent space mean representations
        """
        img = latent_space_visualization_figure(z_mean)
        self.logger.log_image(f'{prefix}/latent_space', images=[img])

    def on_before_optimizer_step(self, optimizer):
        pass
        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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
                "monitor": "val/recon_loss"  # The metric to monitor
            }
        }

    def forward(self, x):
        return self.model(x)


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

        bce_recon_loss = self.binary_cross_entropy(decoded, y)

        self.compute_and_log_metrics(bce_recon_loss, decoded, y, batch_idx, prefix)
        return bce_recon_loss

    def compute_and_log_metrics(self, loss, decoded, y, batch_idx, prefix):
        """
        Compute and log metrics for the current step.

        Args:
            loss: The reconstruction loss for the current step.
            decoded: The model's predictions.
            y: The ground truth sequences.
            batch_idx: The index of the current batch.
            prefix: The prefix for logging (e.g., 'train', 'val', 'test').

        Returns:
            recon_loss: The reconstruction loss for the current step.
        """

        # Log reconstruction loss
        self.log(f'{prefix}/binary_ce_recon_loss', loss, prog_bar=True, sync_dist=True)

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
                "monitor": "val/binary_ce_recon_loss"  # The metric to monitor
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
