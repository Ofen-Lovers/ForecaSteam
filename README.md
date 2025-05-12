# ForecaSteam

A Predictive Modeling of Game Popularity on the Steam Platform Using Game Metadata

## Overview

This project aims to preprocess the Steam Games Dataset to prepare it for machine learning tasks, specifically predicting game popularity based on the 'Estimated owners' metric. The provided Python script (`preprocess.py` - _please rename the script file accordingly_) performs data loading, cleaning, feature engineering, and encoding on the dataset.

## Data Source

The primary dataset used is the **Steam Games Dataset**. Due to its large size, it is not included in this repository. You must download it from Kaggle:

- **Link:** [https://www.kaggle.com/datasets/mexwell/steamgames](https://www.kaggle.com/datasets/mexwell/steamgames)
- **File:** `steam.csv`

**Important:** Download the `steam.csv` file and place it in the same directory as the Python script before running it.

## Requirements

You need Python 3 installed, along with the following libraries:

- pandas
- numpy
- scikit-learn
- scipy
- kagglehub
- joblib
- seaborn
- matplotlib

You can install these using pip:

```bash
pip install pandas numpy scikit-learn scipy kagglehub joblib seaborn matplotlib
```
