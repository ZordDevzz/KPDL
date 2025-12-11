import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes missing or erroneous records from the dataframe.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df

def data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes necessary data and encodes categorical variables for Linear Regression.
    """
    df = df.copy()
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # Normalize numerical columns (excluding target)
    scaler = StandardScaler()
    # Filter numerical columns that are actually in the dataframe and not the target
    cols_to_scale = [col for col in config.NUMERICAL_COLUMNS if col in df.columns and col != config.TARGET_COLUMN]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df
