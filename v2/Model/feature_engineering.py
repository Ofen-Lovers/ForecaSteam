from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
import pandas as pd

def anova_test_numeric(X, y, numeric_cols_for_anova, p_value_threshold=0.05):
    significant_features = []
    non_significant_features = []

    # Ensure numeric_cols_for_anova are present in X
    valid_numeric_cols = [col for col in numeric_cols_for_anova if col in X.columns]
    if not valid_numeric_cols:
        print("\nANOVA: No valid numeric columns found for testing.")
        return X, numeric_cols_for_anova # Return X and original numeric_cols

    # Apply the F-test (ANOVA) to numeric features
    F_values, p_values = f_classif(X[valid_numeric_cols], y)

    anova_results = pd.DataFrame({
        'Feature': valid_numeric_cols,
        'F-value': F_values,
        'p-value': p_values
    })

    updated_numeric_cols = list(valid_numeric_cols) # Start with valid columns

    for feature, p_val in zip(anova_results['Feature'], anova_results['p-value']):
        if p_val < p_value_threshold:
            significant_features.append(feature)
        else:
            non_significant_features.append(feature)
            if feature in X.columns: # Check if column exists before dropping
                X.drop(columns=[feature], inplace=True)
            if feature in updated_numeric_cols: # Also remove from the list
                updated_numeric_cols.remove(feature)


    anova_results = anova_results.sort_values(by='F-value', ascending=False)

    print("\nANOVA Feature Selection Results:")
    print(anova_results)
    print(f"\nSignificant numeric features (p < {p_value_threshold}): {significant_features}")
    print(f"Non-significant numeric features (removed): {non_significant_features}")
    
    return X, updated_numeric_cols # Return X and the updated list of numeric columns

def chi_square_test(df_for_chi, X_for_chi, y_for_chi, p_value_threshold=0.05):
    # df_for_chi is the dataframe with original categorical features before one-hot, target encoded y
    # X_for_chi is the dataframe with one-hot encoded features (where columns will be dropped)
    # y_for_chi is the target variable (label encoded)
    
    # Identify potential categorical/multi-hot columns in X_for_chi
    # These are columns in X_for_chi that are not purely numeric (int/float from original numeric_cols)
    # and are likely one-hot encoded versions of original categorical features
    
    # We need to map original categorical columns to their one-hot encoded versions in X_for_chi
    # For simplicity, we will operate on columns already present in X_for_chi (which are one-hot encoded)
    # and assume 'df_for_chi' contains the original target for contingency table.
    
    # Consider all columns in X_for_chi that are not in the initial numeric_cols list
    # This assumes numeric_cols passed to this function are PRE-ANOVA ones
    
    initial_numeric_cols_in_X = [col for col in X_for_chi.columns if X_for_chi[col].dtype in ['int64', 'float64', 'int32'] and not col.startswith(('Category_', 'Tag_', 'Genre_'))]
    
    # The chi-square test should be applied to binary/categorical features.
    # In X_for_chi, these are the one-hot encoded columns, Windows, Mac, Linux etc.
    # We'll use df_for_chi to get the original values for cross-tabulation for columns that are not one-hot expanded.
    # For one-hot expanded columns (like Category_X, Tag_Y), they are already binary in X_for_chi.

    candidate_chi_features = [
        col for col in X_for_chi.columns 
        if col not in initial_numeric_cols_in_X or col in ['Windows', 'Mac', 'Linux'] # Include known booleans
    ] 
    # Filter out date parts that were numeric but might not be in initial_numeric_cols_in_X if it wasn't updated
    candidate_chi_features = [
        col for col in candidate_chi_features
        if not (col.startswith('Release_date_') or col == 'Game_Age')
    ]


    significant_features = []
    non_significant_features_dropped = []

    print(f"\nChi-Square Test for features (p < {p_value_threshold}):")
    
    # Make a copy of columns to iterate over, as X_for_chi.columns will change
    features_to_test = [col for col in candidate_chi_features if col in X_for_chi.columns]

    for feature in features_to_test:
        if feature not in X_for_chi.columns: # Check if feature was already dropped by a previous iteration
            continue
        try:
            # For one-hot encoded features, X_for_chi[feature] is appropriate
            contingency_table = pd.crosstab(X_for_chi[feature], y_for_chi)
            
            # Check if contingency table is valid (e.g., not all zeros in a row/column)
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2 or contingency_table.sum().sum() == 0:
                # print(f"Skipping Chi-Square for {feature} due to invalid contingency table (e.g. low variance).")
                non_significant_features_dropped.append(feature) # Treat as non-significant if table is bad
                if feature in X_for_chi.columns:
                    X_for_chi.drop(columns=[feature], inplace=True)
                continue

            chi2, p, dof, expected = chi2_contingency(contingency_table)
            
            if p < p_value_threshold:
                significant_features.append(feature)
            else:
                non_significant_features_dropped.append(feature)
                if feature in X_for_chi.columns: # Re-check, as it might be dropped
                     X_for_chi.drop(columns=[feature], inplace=True)
        except ValueError as e:
            print(f"Could not perform Chi-Square test for {feature}: {e}. Dropping feature.")
            non_significant_features_dropped.append(feature)
            if feature in X_for_chi.columns:
                 X_for_chi.drop(columns=[feature], inplace=True)


    print(f"\nSignificant categorical/binary features kept: {len(significant_features)}") # Too many to list all
    # print(", ".join(significant_features))
    print(f"\nNon-significant categorical/binary features dropped: {len(non_significant_features_dropped)}")
    # print(", ".join(non_significant_features_dropped))

    return X_for_chi