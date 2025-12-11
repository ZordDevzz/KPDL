import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
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

def data_transform(df: pd.DataFrame, scaler=None, label_encoder=None, is_predict=False) -> pd.DataFrame:
    """
    Normalizes necessary data and encodes categorical variables for Linear Regression.
    Accepts optional pre-fitted scaler and label_encoder.
    If is_predict is True, it will use the provided (or loaded) transformers without fitting.
    """
    df = df.copy()
    
    # Initialize transformers if not provided
    if scaler is None:
        scaler = StandardScaler()
    if label_encoder is None:
        label_encoder = LabelEncoder()
            
    # Encode categorical columns
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            if is_predict:
                # Use loaded encoder for prediction, handle unseen labels
                # Ensure all categories seen during fit are present
                # Handle cases where unseen labels appear in prediction data
                known_classes = list(label_encoder.classes_)
                df[col] = df[col].apply(lambda x: known_classes.index(x) if x in known_classes else -1) # Assign -1 or other indicator for unseen
            else:
                df[col] = label_encoder.fit_transform(df[col])
            
    # Normalize numerical columns (excluding target)
    cols_to_scale = [col for col in config.NUMERICAL_COLUMNS if col in df.columns and col != config.TARGET_COLUMN]
    
    if cols_to_scale:
        if is_predict:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df, scaler, label_encoder # Return transformers after fitting/transforming

def save_transformers(scaler, label_encoder):
    """
    Saves the fitted StandardScaler and LabelEncoder.
    """
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    print(f"StandardScaler saved to {config.SCALER_PATH}")
    print(f"LabelEncoder saved to {config.LABEL_ENCODER_PATH}")

def load_transformers():
    """
    Loads the fitted StandardScaler and LabelEncoder.
    """
    scaler = joblib.load(config.SCALER_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    print(f"StandardScaler loaded from {config.SCALER_PATH}")
    print(f"LabelEncoder loaded from {config.LABEL_ENCODER_PATH}")
    return scaler, label_encoder
