# Interview Q&A — BengaluruFlow

20 questions at freshman depth with honest, clear answers. Study these
before any technical interview or demo session.

---

### 1. Why did you use LSTM instead of a simple MLP for time series?

An MLP treats each time step as an independent input — it has no built-in memory
of what came before. An LSTM has gated memory cells (input gate, forget gate,
output gate) that let it selectively remember or forget information across time
steps. This makes it much better suited for sequences where the past strongly
influences the future, like traffic patterns that follow daily and weekly cycles.
An MLP would need to see the entire flattened window at once and can't generalise
across different sequence lengths without redesign.

---

### 2. What is a sliding window dataset and how did you implement it?

A sliding window takes a long time series and cuts it into overlapping (input, target)
pairs. For each road, I sorted records chronologically, then slid a window of T=7
consecutive days across the timeline. For each position i, the input is rows [i : i+T]
and the target is rows [i+T : i+T+H]. I implemented this as a PyTorch `Dataset` class
that pre-computes all windows in `__init__` and returns `(X_tensor, y_tensor)` from
`__getitem__`. Windows are built per road so they never span two different roads.

---

### 3. Why chronological split and not random split?

Random splitting would cause **data leakage**: future data ends up in the training
set, and the model can implicitly "see" future patterns during training. This makes
validation and test metrics artificially optimistic and useless for real deployment.
A chronological split (first 70% → train, next 15% → val, last 15% → test) mimics
real-world deployment where you train on historical data and predict future data.

---

### 4. How does the LSTM autoencoder detect anomalies?

The autoencoder is trained only on normal-looking traffic sequences. It learns to
compress a 7-day window into a 32-dimensional bottleneck and reconstruct it back.
Because it was trained on normal data, it reconstructs normal patterns well
(low MSE) but struggles to reconstruct unusual patterns (high MSE). At inference,
I compute the per-sample MSE between the original window and the reconstruction.
If the MSE exceeds the 95th-percentile threshold computed on the validation set,
the window is flagged as anomalous.

---

### 5. What is reconstruction error?

Reconstruction error is the difference between the autoencoder's output and its
input, measured by MSE (mean squared error) averaged over all time steps and
features in the window. Specifically: `MSE = mean((x - x_hat)^2)` where `x` is
the original input and `x_hat` is the reconstructed output. A low MSE means the
model could reconstruct the input well (pattern is familiar). A high MSE means
the pattern was unusual and hard to reconstruct (potential anomaly).

---

### 6. Why fit StandardScaler only on the training set?

If you fit the scaler on the full dataset (or including val/test data), the scaler
"sees" the mean and variance of future data during training. This is **data leakage**.
The model would be trained on statistics derived from data it should never have seen.
By fitting only on training data and then applying the same transformation to val
and test, we simulate real deployment: you compute statistics from historical data
and apply them forward without ever looking at the future.

---

### 7. How does the Transformer differ from LSTM in this context?

The LSTM processes time steps sequentially — it reads day 1, then day 2, ..., 
updating its hidden state one step at a time. The Transformer processes all T=7
time steps simultaneously using **self-attention**, which computes pairwise
relationships between every pair of time steps. This means a Transformer can
directly attend to any past time step without information having to "flow through"
intermediate steps (no vanishing gradient problem over short sequences). However,
Transformers need positional information since they have no inherent notion of
sequence order — I added a learned positional embedding for this.

---

### 8. What is MAPE and when is it misleading?

MAPE = Mean Absolute Percentage Error = `mean(|actual - predicted| / |actual|) * 100`.
It is useful because it is scale-independent (works across different roads with very
different traffic volumes). However, it is **misleading when actual values are near
zero** — a small absolute error like 0.5 becomes a huge percentage error if the
actual is 0.1. In traffic data, roads with very low traffic volume inflate MAPE
disproportionately. That is why I report MAE and RMSE alongside MAPE.

---

### 9. What baselines did you use and why?

I used three:
- **Linear Regression**: The simplest possible model — establishes the floor.
  If a neural network can't beat this, something is wrong.
- **Random Forest**: A strong non-linear ensemble that handles feature interactions
  well without any sequence modelling. A good reference for "how much can tree
  models do without understanding time?"
- **XGBoost**: Gradient-boosted trees — state-of-the-art for tabular data.
  If XGBoost beats the LSTM, it suggests the temporal structure is not providing
  much additional signal, or the DL model needs more tuning.

---

### 10. How did you prevent overfitting?

Three main techniques:
1. **Dropout** (0.2 in LSTM, 0.1 in Transformer): randomly zeroes activations
   during training, preventing the model from relying on any single neuron.
2. **Early stopping** (patience=10): training stops when validation loss hasn't
   improved for 10 consecutive epochs. The best checkpoint is saved and restored.
3. **Train/val/test split**: validation loss guided all hyperparameter choices
   and model selection; test set was touched only for final reporting.

---

### 11. What does MLflow track in your project?

I used basic MLflow tracking, appropriate for a solo project:
- **Params**: learning rate, hidden size, dropout, batch size, window size, horizon, etc.
- **Metrics**: train_mae, val_mae logged per epoch (allowing loss curve inspection
  in the MLflow UI). best_val_mae logged at end of run.
- **Artifacts**: model checkpoint `.pt` files and the scaler `.joblib` are logged,
  making each run self-contained and reproducible.
I did not use MLflow Model Registry — that is more suited to team deployments.

---

### 12. How would you improve this project given more time?

Top 4 improvements:
1. **Hourly data**: The current dataset is daily. Hourly data would reveal
   commute-hour spikes that are more practically useful for traffic management.
2. **Better anomaly labels**: Incident Reports > 0 is a weak proxy. Ground truth
   labels from traffic cameras or police reports would improve evaluation.
3. **Uncertainty estimation**: Add Monte Carlo Dropout to produce confidence
   intervals on forecasts, which is critical for real decision-making.
4. **Spatial context**: Add features that represent the road network neighbourhood
   (e.g., aggregated statistics from nearby roads) to capture how congestion spreads.

---

### 13. What are the limitations of using Incident Reports as anomaly labels?

Several serious ones:
- **False negatives**: Not all anomalous traffic conditions trigger an incident
  report. Traffic jams caused by weather or events may not be reported.
- **False positives**: Minor incidents (e.g., a small fender bender) may be
  reported but not cause anomalous-scale traffic disruption.
- **Temporal misalignment**: An incident might be reported hours after the
  anomalous traffic pattern begins.
- **Sparse labels**: Most rows have 0 incidents, making this a heavily
  class-imbalanced evaluation scenario.

---

### 14. Why did you encode cyclical features as sin/cos?

If you encode day_of_week as the integers 0–6, Sunday (6) and Monday (0) appear
far apart (distance = 6), but they are actually consecutive in the weekly cycle.
This confuses the model. Encoding as `sin(2π * day / 7)` and `cos(2π * day / 7)`
places all days on a circle, so Sunday and Monday are close. The same logic
applies to month encoding. Using both sin and cos ensures the mapping is
invertible (sin alone is ambiguous between, say, day 1 and day 6).

---

### 15. What is early stopping and why use it?

Early stopping monitors validation loss after each epoch and stops training
if it doesn't improve for a set number of epochs (patience). In this project,
patience=10 for LSTM/Transformer and patience=8 for the autoencoder.

Why use it:
1. **Prevents overfitting**: training loss always decreases, but validation loss
   eventually increases as the model memorises training data.
2. **Saves time**: no need to run all 80 epochs if the model converges at epoch 40.
3. **Automatic model selection**: the best checkpoint (lowest val loss) is saved
   automatically — no manual selection needed.

---

### 16. Why did you use MAE (L1) instead of MSE as the forecasting loss?

MAE is more **robust to outliers** than MSE. Traffic data has occasional extreme
values (major incidents, festival days). MSE squares the error, so a single
large spike disproportionately dominates the loss and pulls the model's weights
toward fitting that spike rather than the typical pattern. MAE treats all errors
equally regardless of magnitude, which produces more stable and practically
useful predictions for day-to-day traffic.

---

### 17. What does the bottleneck in the autoencoder do?

The bottleneck (a `Linear(64, 32)` layer) forces the encoder to compress a
7-day traffic window into a 32-dimensional vector. This compression is the key
mechanism: the model must learn the most important patterns in the data to
reconstruct it from only 32 numbers. Normal patterns are common and easy to
learn; anomalous patterns are rare and can't be stored efficiently in the
compressed representation, leading to higher reconstruction error.

---

### 18. What is the difference between val set and test set, and why have both?

- **Validation set**: used *during* training to monitor generalisation, guide
  early stopping, and tune hyperparameters (hidden size, learning rate, etc.).
  Any decision made by looking at val loss has "seen" the val set indirectly.
- **Test set**: touched only once, after all model selection is done, for
  final honest performance reporting.

If you used the val set for final reporting, you risk optimistically biased
metrics because the hyperparameters were chosen to perform well on it. The test
set is the only unbiased estimate of real-world performance.

---

### 19. How does ReduceLROnPlateau work, and why did you use it for the LSTM?

`ReduceLROnPlateau` monitors a tracked metric (val_loss in this project) and
reduces the learning rate by a factor (0.5) when the metric stops improving for
`patience=5` epochs. This is useful because:
- The optimal learning rate changes throughout training — a high lr helps early,
  but may cause oscillations near the optimum.
- Reducing the lr allows finer updates near convergence, squeezing out extra
  performance without manual learning rate scheduling.
I applied it to the LSTM but not the Transformer, since the Transformer used a
fixed smaller lr (5e-4) and gradient clipping for stability.

---

### 20. If the LSTM and Transformer give similar MAE, which would you deploy?

For this dataset (daily, ~9K rows, short T=7 sequences), I would likely deploy
the LSTM because:
1. **Simpler to debug**: no attention masks, positional encoding, or multi-head
   complexity to reason about.
2. **Faster at inference**: a 2-layer LSTM with 128 hidden units is computationally
   cheaper than a Transformer for single-sample real-time predictions.
3. **Better studied for short sequences**: self-attention shines on very long
   sequences. At T=7, the LSTM has no memory bottleneck.

If the dataset were hourly with T=168 (one week of hours), the Transformer's
ability to directly attend across any time lag would likely make it superior.
