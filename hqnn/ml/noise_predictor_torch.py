import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class LSTMAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.layer_norm(lstm_out + self.dropout(attn_out))
        return out, attn_weights


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)) + self.residual(x))


class NoisePredictor(nn.Module):
    def __init__(self,
                 seq_length: int = 20,
                 n_features: int = 5,
                 hidden_dim: int = 128,
                 n_lstm_layers: int = 2,
                 n_heads: int = 4,
                 forecast_horizon: int = 5):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.input_projection = nn.Linear(n_features, hidden_dim)

        self.tcn_blocks = nn.Sequential(
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=1),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=2),
            TemporalConvBlock(hidden_dim, hidden_dim, dilation=4),
        )

        self.lstm_attention = nn.ModuleList([
            LSTMAttentionBlock(hidden_dim, hidden_dim // 2, n_heads)
            for _ in range(n_lstm_layers)
        ])

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, forecast_horizon),
            nn.Softplus(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, forecast_horizon),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_projection(x)

        tcn_in = projected.transpose(1, 2)
        tcn_out = self.tcn_blocks(tcn_in)
        hidden = tcn_out.transpose(1, 2)

        attn_weights_all = []
        for lstm_attn in self.lstm_attention:
            hidden, attn_w = lstm_attn(hidden)
            attn_weights_all.append(attn_w)

        pooled = hidden.mean(dim=1)

        predictions = self.output_head(pooled)
        uncertainty = self.uncertainty_head(pooled)

        return predictions, uncertainty


class NoisePredictorTrainer:
    def __init__(self,
                 model: NoisePredictor,
                 device: str = "auto",
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(),
                                      lr=learning_rate,
                                      weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def prepare_data(self, noise_sequence: np.ndarray,
                      extra_features: Optional[np.ndarray] = None,
                      val_split: float = 0.15) -> Tuple[DataLoader, DataLoader]:
        seq_len = self.model.seq_length
        horizon = self.model.forecast_horizon
        n_features = self.model.n_features

        if extra_features is None:
            diff1 = np.diff(noise_sequence, n=1, prepend=noise_sequence[0])
            diff2 = np.diff(noise_sequence, n=2, prepend=noise_sequence[:2])
            rolling_mean = np.convolve(noise_sequence,
                                        np.ones(5) / 5, mode='same')
            rolling_std = np.array([
                np.std(noise_sequence[max(0, i-5):i+1])
                for i in range(len(noise_sequence))
            ])
            features = np.stack([noise_sequence, diff1, diff2,
                                  rolling_mean, rolling_std], axis=1)
        else:
            features = extra_features

        X, y = [], []
        for i in range(len(features) - seq_len - horizon):
            X.append(features[i:i + seq_len])
            y.append(noise_sequence[i + seq_len:i + seq_len + horizon])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        n_val = int(len(X) * val_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        train_ds = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val),
                                torch.from_numpy(y_val))

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                   num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                 num_workers=0, pin_memory=True)

        return train_loader, val_loader

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds, uncertainty = self.model(X_batch)

            mse_loss = nn.functional.mse_loss(preds, y_batch)
            nll_loss = (torch.log(uncertainty + 1e-6) +
                        (y_batch - preds) ** 2 / (2 * uncertainty ** 2 + 1e-6)).mean()
            loss = mse_loss + 0.1 * nll_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def validate_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds, _ = self.model(X_batch)
                loss = nn.functional.mse_loss(preds, y_batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(1, n_batches)

    def train(self, noise_sequence: np.ndarray,
              n_epochs: int = 100,
              early_stopping_patience: int = 15,
              checkpoint_path: Optional[str] = None) -> Dict:
        train_loader, val_loader = self.prepare_data(noise_sequence)
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_path:
                    torch.save({
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    }, checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(self.train_losses),
            "device": str(self.device),
        }

    def predict(self, noise_buffer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        seq_len = self.model.seq_length

        if len(noise_buffer) < seq_len:
            padding = np.zeros(seq_len - len(noise_buffer), dtype=np.float32)
            noise_buffer = np.concatenate([padding, noise_buffer])

        seq = noise_buffer[-seq_len:].astype(np.float32)
        diff1 = np.diff(seq, n=1, prepend=seq[0])
        diff2 = np.diff(seq, n=2, prepend=seq[:2])
        rolling_mean = np.convolve(seq, np.ones(5) / 5, mode='same')
        rolling_std = np.array([np.std(seq[max(0, i-5):i+1]) for i in range(len(seq))])

        features = np.stack([seq, diff1, diff2, rolling_mean, rolling_std], axis=1)
        X = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds, uncertainty = self.model(X)

        return preds.cpu().numpy()[0], uncertainty.cpu().numpy()[0]