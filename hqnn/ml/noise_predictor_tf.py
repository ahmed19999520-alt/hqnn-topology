import numpy as np
import os
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks


def build_tcn_block(x: tf.Tensor, filters: int,
                     kernel_size: int, dilation: int) -> tf.Tensor:
    padding = dilation * (kernel_size - 1)
    x_pad = tf.pad(x, [[0, 0], [padding, 0], [0, 0]])
    conv_out = layers.Conv1D(filters, kernel_size, dilation_rate=dilation,
                              padding="valid", activation="gelu")(x_pad)
    conv_out = layers.BatchNormalization()(conv_out)
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1)(x)
    return layers.Add()([conv_out, x])


def build_transformer_block(x: tf.Tensor, d_model: int,
                              n_heads: int, ff_dim: int,
                              dropout: float = 0.1) -> tf.Tensor:
    attn_output = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ff_output = layers.Dense(ff_dim, activation="gelu")(x)
    ff_output = layers.Dropout(dropout)(ff_output)
    ff_output = layers.Dense(d_model)(ff_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
    return x


def build_noise_predictor_tf(seq_length: int = 20,
                               n_features: int = 5,
                               d_model: int = 128,
                               n_heads: int = 4,
                               n_transformer_blocks: int = 3,
                               forecast_horizon: int = 5) -> Model:
    inputs = keras.Input(shape=(seq_length, n_features), name="noise_sequence")

    x = layers.Dense(d_model, name="input_projection")(inputs)

    x = build_tcn_block(x, d_model, kernel_size=3, dilation=1)
    x = build_tcn_block(x, d_model, kernel_size=3, dilation=2)
    x = build_tcn_block(x, d_model, kernel_size=3, dilation=4)

    for i in range(n_transformer_blocks):
        x = build_transformer_block(x, d_model, n_heads, d_model * 2)

    lstm_out = layers.Bidirectional(
        layers.LSTM(d_model // 2, return_sequences=False))(x)

    pooled = layers.GlobalAveragePooling1D()(x)
    combined = layers.Concatenate()([lstm_out, pooled])
    combined = layers.LayerNormalization()(combined)

    pred_head = layers.Dense(d_model // 2, activation="gelu")(combined)
    pred_head = layers.Dropout(0.1)(pred_head)
    pred_head = layers.Dense(d_model // 4, activation="gelu")(pred_head)
    predictions = layers.Dense(forecast_horizon, activation="softplus",
                                name="predictions")(pred_head)

    unc_head = layers.Dense(d_model // 4, activation="gelu")(combined)
    uncertainty = layers.Dense(forecast_horizon, activation="softplus",
                                name="uncertainty")(unc_head)

    model = Model(inputs=inputs, outputs=[predictions, uncertainty],
                  name="NoisePredictorTF")
    return model


class NoisePredictorTFTrainer:
    def __init__(self,
                 seq_length: int = 20,
                 n_features: int = 5,
                 d_model: int = 128,
                 n_heads: int = 4,
                 forecast_horizon: int = 5,
                 learning_rate: float = 1e-3):
        self.seq_length = seq_length
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon

        self.model = build_noise_predictor_tf(
            seq_length=seq_length,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            forecast_horizon=forecast_horizon,
        )

        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def _gaussian_nll_loss(self, y_true: tf.Tensor,
                            y_pred: tf.Tensor,
                            sigma: tf.Tensor) -> tf.Tensor:
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        nll = tf.reduce_mean(
            tf.math.log(sigma + 1e-6) +
            tf.square(y_true - y_pred) / (2 * tf.square(sigma) + 1e-6)
        )
        return mse + 0.1 * nll

    def prepare_dataset(self, noise_sequence: np.ndarray,
                         val_split: float = 0.15) -> Tuple:
        seq_len = self.seq_length
        horizon = self.forecast_horizon

        diff1 = np.diff(noise_sequence, n=1, prepend=noise_sequence[0])
        diff2 = np.diff(noise_sequence, n=2, prepend=noise_sequence[:2])
        rolling_mean = np.convolve(noise_sequence, np.ones(5) / 5, mode="same")
        rolling_std = np.array([
            np.std(noise_sequence[max(0, i-5):i+1])
            for i in range(len(noise_sequence))
        ])
        features = np.stack([noise_sequence, diff1, diff2,
                              rolling_mean, rolling_std], axis=1).astype(np.float32)

        X, y = [], []
        for i in range(len(features) - seq_len - horizon):
            X.append(features[i:i + seq_len])
            y.append(noise_sequence[i + seq_len:i + seq_len + horizon])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        n_val = int(len(X) * val_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    .shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE))
        val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                  .batch(64).prefetch(tf.data.AUTOTUNE))

        return train_ds, val_ds

    @tf.function
    def train_step(self, X_batch: tf.Tensor,
                    y_batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            preds, uncertainty = self.model(X_batch, training=True)
            loss = self._gaussian_nll_loss(y_batch, preds, uncertainty)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, noise_sequence: np.ndarray,
              n_epochs: int = 100,
              checkpoint_dir: Optional[str] = None,
              early_stopping_patience: int = 15) -> Dict:
        train_ds, val_ds = self.prepare_dataset(noise_sequence)
        best_val = float("inf")
        patience_counter = 0

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

        for epoch in range(n_epochs):
            train_loss_sum = 0.0
            n_batches = 0
            for X_batch, y_batch in train_ds:
                loss = self.train_step(X_batch, y_batch)
                train_loss_sum += float(loss.numpy())
                n_batches += 1

            train_loss = train_loss_sum / max(1, n_batches)

            val_loss_sum = 0.0
            n_val = 0
            for X_val, y_val in val_ds:
                preds, _ = self.model(X_val, training=False)
                val_loss_sum += float(tf.reduce_mean(
                    tf.square(y_val - preds)).numpy())
                n_val += 1
            val_loss = val_loss_sum / max(1, n_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                if checkpoint_dir:
                    ckpt_manager.save()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": best_val,
            "epochs_trained": len(self.train_losses),
        }

    def predict(self, noise_buffer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = self.seq_length
        if len(noise_buffer) < seq_len:
            padding = np.zeros(seq_len - len(noise_buffer), dtype=np.float32)
            noise_buffer = np.concatenate([padding, noise_buffer])

        seq = noise_buffer[-seq_len:].astype(np.float32)
        diff1 = np.diff(seq, n=1, prepend=seq[0])
        diff2 = np.diff(seq, n=2, prepend=seq[:2])
        rolling_mean = np.convolve(seq, np.ones(5) / 5, mode="same")
        rolling_std = np.array([np.std(seq[max(0, i-5):i+1]) for i in range(len(seq))])
        features = np.stack([seq, diff1, diff2, rolling_mean, rolling_std], axis=1)
        X = tf.constant(features[np.newaxis], dtype=tf.float32)

        preds, uncertainty = self.model(X, training=False)
        return preds.numpy()[0], uncertainty.numpy()[0]

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = keras.models.load_model(path)