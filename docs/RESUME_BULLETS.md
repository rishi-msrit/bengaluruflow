# Resume Bullets — BengaluruFlow

Eight bullet points for your resume. Start with strong action verbs,
include specific techniques and numbers, and stay honest about scope.

---

**BengaluruFlow: Traffic Forecasting & Anomaly Detection** | PyTorch · MLflow · Streamlit

- **Built** a multi-output LSTM forecaster in PyTorch that predicts Traffic Volume,
  Congestion Level, and Average Speed simultaneously for H=3 day horizons using a
  T=7 day look-back window on 8,936 real Bangalore traffic records across 16 features.

- **Implemented** a Transformer-based sequence forecaster from scratch using
  `nn.TransformerEncoder` with learned positional embeddings, comparing it against
  the LSTM on MAE, RMSE, MAPE, and R² across three target variables.

- **Engineered** 9+ temporal features (cyclical sin/cos day/month encodings,
  is_weekend, is_holiday, days_since_start) and applied StandardScaler fitted
  exclusively on the training split to avoid data leakage.

- **Designed** an LSTM sequence autoencoder for unsupervised anomaly detection,
  using a 32-dimensional bottleneck to compress T=7 traffic windows and flagging
  samples whose reconstruction MSE exceeds the 95th-percentile validation threshold.

- **Benchmarked** deep learning models against three scikit-learn baselines
  (Linear Regression, Random Forest, XGBoost) using a chronological 70/15/15
  train/val/test split, demonstrating measurable MAE improvement of the best DL
  model over the Linear Regression baseline.

- **Tracked** all hyperparameters and metrics across four MLflow experiments
  (lstm_forecaster, transformer_forecaster, baseline_models, anomaly_detector),
  enabling reproducible experiment comparison via the MLflow UI.

- **Developed** a 4-page interactive Streamlit demo that loads models with
  `@st.cache_resource`, handles missing artifacts gracefully, and allows
  on-demand LSTM inference for any road in the dataset with a live line chart output.

- **Wrote** 26 pytest unit tests covering CSV column validation, feature engineering
  correctness (sin/cos bounds, binary constraints), sliding-window dataset shapes,
  and forward-pass output shapes for all three neural network architectures.
