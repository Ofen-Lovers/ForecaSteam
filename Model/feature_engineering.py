from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
import pandas as pd

def anova_test_numeric(numeric_cols, X, y):
    # Lists to store significant and non-significant features
    significant_features = []
    non_significant_features = []

    # Apply the F-test (ANOVA) to numeric features
    F_values, p_values = f_classif(X[numeric_cols], y)

    # Create a DataFrame of features with their F-statistics and p-values
    anova_results = pd.DataFrame({
        'Feature': numeric_cols,
        'F-value': F_values,
        'p-value': p_values
    })

    # Classify features based on p-value
    for feature, p in zip(anova_results['Feature'], anova_results['p-value']):
        if p < 0.05:
            significant_features.append(feature)
        else:
            non_significant_features.append(feature)
            X.drop(columns=[feature], inplace=True)

    # Sort by F-value to determine the most important features
    anova_results = anova_results.sort_values(by='F-value', ascending=False)

    # Print Result
    print("\nANOVA Feature Selection Results:")
    print(anova_results)

    print(f"\nSignificant features: {significant_features}")
    print(f"Non-significant features: {non_significant_features}")
    print(f"\nRevomved Feature/s: {non_significant_features}")

    return X

def chi_square_test(df, target, numeric_cols, X):
    # Multi-hot columns
    multi_hot_cols = [col for col in df.columns if col not in numeric_cols and col != target]
    
    # category_cols = [col for col in df.columns if col.startswith('Category')]
    # tag_cols = [col for col in df.columns if col.startswith('Tag')]
    # genre_cols = [col for col in df.columns if col.startswith('Genre')]
    # audio_cols = [col for col in df.columns if col.startswith('Audio')]
    # lang_cols = [col for col in df.columns if col.startswith('Lang')]
    
    # Lists to store significant and non-significant features
    significant_features = []
    non_significant_features = []

    for feature in multi_hot_cols:  # You can change this to loop through other lists like category_cols, etc.
        contingency_table = pd.crosstab(df[feature], df[target])
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # print(f"Chi-Square Test for {feature}:")
        # print(f"Chi2 statistic: {chi2}, p-value: {p}")
        
        if p < 0.05:
            significant_features.append(feature)  
            
        else:
            non_significant_features.append(feature)
            X.drop(columns=[feature], inplace=True)

    # Print results
    print("Features that are significantly related to the target:")
    print(", ".join(significant_features))

    print("\nFeatures that are not significantly related to the target:")
    print(", ".join(non_significant_features))

    print("\nFeatures dropped:")
    print(", ".join(non_significant_features))

    return X


