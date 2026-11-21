"""Dataset profiler for ModelDoctor."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Profile a tabular dataset and return a JSON-serializable dictionary
    summarizing dimensions, data types, missingness, class balance,
    and data quality warnings.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    target_column : str
        The target column name.

    Returns
    -------
    dict
        JSON-serializable profile report.
    """
    row_count = int(len(df))
    column_count = int(len(df.columns))

    columns_info: List[Dict[str, Any]] = []
    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    warnings: List[str] = []

    target_series = df[target_column]

    for col in df.columns:
        series = df[col]
        is_num = bool(pd.api.types.is_numeric_dtype(series))
        col_type = "numeric" if is_num else "categorical"
        missing_count = int(series.isna().sum())
        missing_pct = round(float(missing_count / row_count * 100.0), 2) if row_count > 0 else 0.0
        unique_count = int(series.nunique(dropna=False))

        is_target = bool(col == target_column)
        if not is_target:
            if is_num:
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)

            # Warning checks on feature columns
            # 1. Constant columns
            if unique_count <= 1:
                warnings.append(
                    f"Column '{col}' is constant (has only {unique_count} unique value) and provides no predictive signal."
                )

            # 2. Identifier-like columns. Continuous float features are naturally
            # all-unique, so only integer / string columns are treated as ID-like.
            elif pd.api.types.is_float_dtype(series):
                pass
            elif unique_count == row_count:
                warnings.append(
                    f"Column '{col}' has 100% unique values ({unique_count} distinct) and may be an ID column."
                )
            elif row_count >= 100 and (unique_count / row_count > 0.95):
                warnings.append(
                    f"Column '{col}' has {unique_count} distinct values ({round((unique_count/row_count)*100, 1)}% unique) and may be an ID column."
                )

            # 3. High cardinality categorical columns
            elif not is_num and unique_count > 50:
                warnings.append(
                    f"Categorical column '{col}' has high cardinality ({unique_count} unique categories)."
                )

            # 4. High missingness columns
            if missing_pct >= 50.0:
                warnings.append(
                    f"Column '{col}' has {missing_pct}% missing values ({missing_count}/{row_count} rows)."
                )

        columns_info.append({
            "name": str(col),
            "dtype": col_type,
            "raw_dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_count": unique_count,
            "is_target": is_target,
        })

    # Class balance and distribution
    class_counts_raw = target_series.value_counts(dropna=False)
    class_distribution: Dict[str, int] = {
        str(k): int(v) for k, v in class_counts_raw.items()
    }
    class_percentages: Dict[str, float] = {
        str(k): round(float((v / row_count) * 100.0), 2) for k, v in class_counts_raw.items()
    }

    min_class_count = int(class_counts_raw.min()) if len(class_counts_raw) > 0 else 1
    max_class_count = int(class_counts_raw.max()) if len(class_counts_raw) > 0 else 1
    imbalance_ratio = round(float(max_class_count / max(min_class_count, 1)), 2)

    if imbalance_ratio >= 10.0:
        warnings.append(
            f"Severe class imbalance detected: majority class is {imbalance_ratio}x larger than minority class."
        )

    num_classes = len(class_counts_raw)
    detected_task = "binary_classification" if num_classes == 2 else "multiclass_classification"

    target_dtype = "numeric" if pd.api.types.is_numeric_dtype(target_series) else "categorical"

    return {
        "row_count": row_count,
        "column_count": column_count,
        "feature_count": column_count - 1,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "columns": columns_info,
        "target_column": str(target_column),
        "target_dtype": target_dtype,
        "detected_task": detected_task,
        "class_distribution": class_distribution,
        "class_percentages": class_percentages,
        "imbalance_ratio": imbalance_ratio,
        "warnings": warnings,
    }
