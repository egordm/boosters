# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Documentation**: Complete Sphinx documentation infrastructure
  - Getting started guides for Python and Rust
  - 9 Jupyter notebook tutorials
  - Comprehensive API reference
  - Explanations for gradient boosting, hyperparameters, and objectives
  - How-to guides for common tasks
  - GitHub Actions workflow for docs deployment

- **Model Serialization**: Pickle support for Python models
  - `GBDTModel` and `GBLinearModel` now support `pickle.dump()` and `pickle.load()`
  - Uses native binary format internally for efficiency

### Changed

- N/A

### Deprecated

- N/A

### Removed

- N/A

### Fixed

- N/A

### Security

- N/A

## [0.1.0] - Unreleased

Initial release with core functionality:

- **Training**
  - GBDT training with histogram-based algorithm
  - GBLinear training with coordinate descent
  - GOSS (Gradient-based One-Side Sampling)
  - Row and column subsampling

- **Inference**
  - Fast batch prediction
  - Single-row prediction
  - SHAP value computation

- **Objectives**
  - Regression: squared, absolute, Huber, quantile, Poisson
  - Classification: logistic, softmax
  - Ranking: pairwise, LambdaRank

- **Metrics**
  - RMSE, MAE, MAPE
  - Log-loss, accuracy, AUC
  - NDCG, MAP

- **Model I/O**
  - Native binary format (.bstr)
  - Native JSON format (.bstr.json)
  - XGBoost/LightGBM model conversion

- **Python Bindings**
  - Core API with `GBDTModel`, `GBLinearModel`, `Dataset`
  - sklearn-compatible estimators
  - Type stubs for IDE support
