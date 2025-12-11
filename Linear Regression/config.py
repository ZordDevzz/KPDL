import os

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "Student_Performance.csv")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_student_performance.csv")
TRANFORMED_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR, "transformed_student_performance.csv"
)

EDA_DIR = os.path.join(PROJECT_ROOT, "eda")
FIGURES_DIR = os.path.join(EDA_DIR, "figures")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Transformer paths
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Ensure all directories exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EDA_DIR,
    FIGURES_DIR,
    MODEL_DIR,
    EVAL_DIR,
    LOG_DIR,
]:
    os.makedirs(directory, exist_ok=True)

# Configure column
TARGET_COLUMN = "Performance Index"
NUMERICAL_COLUMNS = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
CATEGORICAL_COLUMNS = ["Extracurricular Activities"]
