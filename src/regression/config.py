from pathlib import Path

from src.config import SEED



PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = DATA_DIR / "train_processed.parquet"
VAL_PATH = DATA_DIR / "val_processed.parquet"
TEST_PATH = DATA_DIR / "test_processed.parquet"

PREPROCESSOR_PATH = PROJECT_ROOT / "src" / "preprocessing" / "preprocessor.pkl"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "regression"


# --------------------------------------------------
# Regression configuration
# --------------------------------------------------

TARGET = "ADR"

DROP_COLUMNS = [
    "IsCanceled",
    "AssignedRoomType_A",
    "AssignedRoomType_B",
    "AssignedRoomType_C",
    "AssignedRoomType_D",
    "AssignedRoomType_E",
    "AssignedRoomType_F",
    "AssignedRoomType_G",
    "AssignedRoomType_K",
    "AssignedRoomType_P",
    "BookingChanges",
    "DaysInWaitingList",
    "ReservationStatus_Canceled",
    "ReservationStatus_Check-Out",
    "ReservationStatus_No-Show",
    "ReservationStatusDate",
]

MODEL_NAME = "linear_regression"

MODEL_CONFIG = {}
