import pandas as pd
import numpy as np
from sklearn.datasets import load_wine


def test_data_schema():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Check column names
    expected_cols = {'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                     'magnesium', 'total_phenols', 'flavanoids',
                     'nonflavanoid_phenols', 'proanthocyanins',
                     'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                     'proline'}
    assert set(df.columns) == expected_cols

    # Check data types
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    # Check target values
    assert set(data.target) == {0, 1, 2}


def test_data_quality():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Check for missing values
    assert df.isnull().sum().sum() == 0

    # Check value ranges
    assert df['alcohol'].between(11, 15).all()
    assert df['proline'].between(200, 1700).all()

    # Check class distribution
    from collections import Counter
    class_counts = Counter(data.target)
    for count in class_counts.values():
        assert count > 40, "Class imbalance detected"