# ForecaSteam

A machine learning project that predicts game popularity on Steam using game metadata. The project has evolved through multiple versions, with v4 being the current implementation using a Random Forest Classifier.

## Quick Start

1. **Setup**
   ```bash
   # Install dependencies
   pip install pandas numpy scikit-learn scipy kagglehub joblib seaborn matplotlib
   
   # Download dataset
   # Place steam.csv in the Data/ directory
   ```

2. **Run**
   ```bash
   # For version 4 (recommended)
   cd v4
   python main.py
   
   # For version 1
   cd v1
   python main.py
   ```

## Project Structure
```
ForecaSteam/
├── Data/                  # Raw data directory
│   └── steam.csv         # Steam games dataset
├── v1/                   # Initial regression model
│   ├── images/           # Generated visualizations
│   ├── Model/            # Core ML components
│   ├── pkl/              # Saved model artifacts
│   ├── Predictions/      # Prediction scripts
│   └── Processed_Data/   # Preprocessed data
└── v4/                   # Current classification model
    ├── images/           # Generated visualizations
    ├── Model/            # Core ML components
    ├── pkl/              # Saved model artifacts
    ├── Predictions/      # Prediction scripts
    └── Processed_Data/   # Preprocessed data
```

## Features

### Data Processing
- Automated data cleaning and preprocessing
- Feature engineering for game metadata
- Handling of missing values and outliers
- Normalization and scaling of numeric features

### Model Capabilities
- Classification of game popularity ranges
- Feature importance analysis
- Cross-validation and hyperparameter tuning
- Comprehensive model evaluation metrics

### Outputs
- Trained model artifacts
- Processed datasets
- Visualization plots
- Prediction results

## Version Comparison

### Version 4 (Current)
- **Type**: Classification
- **Model**: Random Forest Classifier
- **Performance**:
  - Accuracy: 82%
  - F1 Score: 0.80
  - CV Accuracy: 81.57%
- **Key Features**:
  - Handles imbalanced classes
  - Detailed classification metrics
  - Confusion matrix visualization
  - Feature importance plots

### Version 1
- **Type**: Regression
- **Model**: Random Forest Regressor
- **Performance**:
  - MSE: 3.71
  - R² Score: 0.528
  - CV MSE: 3.71
- **Key Features**:
  - Basic preprocessing
  - Initial feature engineering
  - Regression metrics

## Data Source

The project uses the [Steam Games Dataset](https://www.kaggle.com/datasets/mexwell/steamgames) from Kaggle. The dataset includes:
- Game metadata (price, DLC, etc.)
- User statistics (reviews, playtime)
- Technical details (platforms, languages)
- Release information

## Requirements

- Python 3.x
- Core ML libraries (pandas, numpy, scikit-learn)
- Visualization tools (matplotlib, seaborn)
- Data handling (scipy, joblib)
- Kaggle integration (kagglehub)