from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
import pandas as pd

def anova_test_numeric(X: pd.DataFrame, y: pd.Series, 
                       numeric_cols_for_anova: list, 
                       p_value_threshold: float = 0.05) -> tuple[pd.DataFrame, list]:
    """
    Performs ANOVA F-test for numeric features against a target variable.
    Drops non-significant features from X and returns the updated X
    and the list of remaining significant numeric columns.

    Args:
        X (pd.DataFrame): DataFrame containing features.
        y (pd.Series): Target variable.
        numeric_cols_for_anova (list): List of numeric column names in X to test.
        p_value_threshold (float): Significance level.

    Returns:
        tuple[pd.DataFrame, list]: Updated DataFrame X and list of significant numeric columns.
    """
    significant_features = []
    non_significant_features_removed = []

    valid_numeric_cols = [col for col in numeric_cols_for_anova if col in X.columns]
    if not valid_numeric_cols:
        print("\nANOVA: No valid numeric columns found for testing.")
        return X, numeric_cols_for_anova 

    F_values, p_values = f_classif(X[valid_numeric_cols], y)

    anova_results = pd.DataFrame({
        'Feature': valid_numeric_cols,
        'F-value': F_values,
        'p-value': p_values
    })

    updated_numeric_cols = list(valid_numeric_cols) 

    for feature, p_val in zip(anova_results['Feature'], anova_results['p-value']):
        if p_val < p_value_threshold:
            significant_features.append(feature)
        else:
            non_significant_features_removed.append(feature)
            if feature in X.columns:
                X.drop(columns=[feature], inplace=True)
            if feature in updated_numeric_cols:
                updated_numeric_cols.remove(feature)

    anova_results = anova_results.sort_values(by='F-value', ascending=False)

    print("\nANOVA Feature Selection Results:")
    print(anova_results)
    print(f"\nSignificant numeric features (p < {p_value_threshold}): {significant_features}")
    print(f"Non-significant numeric features (removed): {non_significant_features_removed}")
    
    return X, updated_numeric_cols

def chi_square_test(X_features: pd.DataFrame, y_target: pd.Series, 
                    numeric_cols_to_exclude: list, 
                    p_value_threshold: float = 0.05) -> pd.DataFrame:
    """
    Performs Chi-Square test for categorical/binary features against a target variable.
    Drops non-significant features from X_features.

    Args:
        X_features (pd.DataFrame): DataFrame containing features (already one-hot encoded).
        y_target (pd.Series): Target variable (label encoded).
        numeric_cols_to_exclude (list): List of numeric column names (e.g., from ANOVA) 
                                         that should not be subjected to Chi-Square test.
        p_value_threshold (float): Significance level.

    Returns:
        pd.DataFrame: Updated DataFrame X_features.
    """
    
    # Candidate features for Chi-Square are those in X_features not in numeric_cols_to_exclude
    candidate_chi_features = [
        col for col in X_features.columns if col not in numeric_cols_to_exclude
    ]
    
    significant_features_kept = []
    non_significant_features_dropped = []

    print(f"\nChi-Square Test for features (p < {p_value_threshold}):")
    
    # Iterate over a copy, as X_features.columns will change if columns are dropped
    features_to_test = [col for col in candidate_chi_features if col in X_features.columns]

    for feature in features_to_test:
        if feature not in X_features.columns: # Check if feature was already dropped
            continue
        try:
            contingency_table = pd.crosstab(X_features[feature], y_target)
            
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2 or \
               contingency_table.sum().sum() == 0 or \
               (contingency_table == 0).all(axis=None): # Check if all values are zero
                # print(f"Skipping Chi-Square for {feature} due to invalid contingency table (e.g., low variance).")
                non_significant_features_dropped.append(feature)
                if feature in X_features.columns:
                    X_features.drop(columns=[feature], inplace=True)
                continue

            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            if p < p_value_threshold:
                significant_features_kept.append(feature)
            else:
                non_significant_features_dropped.append(feature)
                if feature in X_features.columns:
                     X_features.drop(columns=[feature], inplace=True)
        except ValueError as e:
            print(f"Could not perform Chi-Square test for {feature}: {e}. Dropping feature.")
            non_significant_features_dropped.append(feature)
            if feature in X_features.columns:
                 X_features.drop(columns=[feature], inplace=True)

    print(f"\nSignificant categorical/binary features kept: {len(significant_features_kept)}")
    print(f"Non-significant categorical/binary features dropped: {len(non_significant_features_dropped)}")

    return X_features