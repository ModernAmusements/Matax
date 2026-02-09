# NGO Facial Image Analysis System - Configuration Template

# General Settings
APP_NAME = "NGO Facial Image Analysis System"
APP_VERSION = "0.1.0"
DEBUG_MODE = False
LOG_LEVEL = "INFO"

# Face Detection Settings
FACE_DETECTION_MODEL = "deploy.prototxt.txt"
FACE_DETECTION_WEIGHTS = "res10_300x300_ssd_iter_140000.caffemodel"
DETECTION_CONFIDENCE_THRESHOLD = 0.5
DETECTION_IMAGE_SIZE = (300, 300)

# Embedding Extraction Settings
EMBEDDING_MODEL_PATH = "facenet_model.pth"
EMBEDDING_SIZE = 128
EMBEDDING_NORMALIZATION = True

# Similarity Comparison Settings
SIMILARITY_THRESHOLD = 0.6
MIN_CONFIDENCE_SCORE = 0.4
MAX_CONFIDENCE_SCORE = 0.8

# Reference Image Settings
REFERENCE_IMAGE_DIR = "reference_images"
EMBEDDINGS_FILE = "embeddings.json"
MAX_REFERENCE_IMAGES = 1000

# Human Review Settings
REVIEW_CONFIDENCE_BANDS = {
    "HIGH": 0.8,
    "MODERATE": 0.6,
    "LOW": 0.4,
    "INSUFFICIENT": 0.0
}
REVIEW_HISTORY_ENABLED = True
REVIEW_AUTO_SAVE = True

# Webcam Settings
WEBCAM_INDEX = 0
WEBCAM_FRAME_WIDTH = 640
WEBCAM_FRAME_HEIGHT = 480
WEBCAM_FPS = 30
WEBCAM_DETECTION_ENABLED = True
WEBCAM_DISPLAY_ENABLED = True

# File Paths
LOG_FILE = "logs/face_recognition.log"
TEMP_DIR = "temp"
MODEL_DIR = "models"
DATA_DIR = "data"

# Security Settings
ENCRYPTED_STORAGE = False
DATA_RETENTION_DAYS = 30
BACKUP_ENABLED = True

# Performance Settings
BATCH_SIZE = 8
PROCESSING_THREADS = 4
GPU_ACCELERATION = False
CACHE_ENABLED = True

# NGO-Specific Settings
CONSENT_REQUIRED = True
AUDIT_LOGGING_ENABLED = True
DATA_ANONYMIZATION = True
PRIVACY_COMPLIANCE = "GDPR"

# Error Handling Settings
MAX_RETRY_ATTEMPTS = 3
TIMEOUT_SECONDS = 30
FALLBACK_MODE = False

# Testing Settings
TEST_IMAGE_DIR = "test_images"
TEST_REFERENCE_DIR = "test_references"
TEST_WEBCAM_ENABLED = False

# Documentation Settings
DOCUMENTATION_DIR = "docs"
EXAMPLE_DIR = "examples"
HELP_FILE = "README.md"

# Network Settings
API_TIMEOUT = 10
MAX_CONNECTIONS = 5
PROXY_ENABLED = False

# Logging Settings
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_ROTATION = True
LOG_MAX_SIZE = "10MB"

# Development Settings
DEV_MODE = False
VERBOSE_OUTPUT = False
PROFILING_ENABLED = False

# NGO Compliance Settings
HUMAN_OVERSIGHT_REQUIRED = True
AUTOMATED_IDENTIFICATION_DISABLED = True
ETHICAL_GUIDELINES_ENABLED = True

# Default paths (will be created if they don't exist)
PATHS = {
    "log_dir": LOG_FILE.rsplit('/', 1)[0],
    "temp_dir": TEMP_DIR,
    "model_dir": MODEL_DIR,
    "data_dir": DATA_DIR,
    "reference_dir": REFERENCE_IMAGE_DIR,
    "test_dir": TEST_IMAGE_DIR,
    "doc_dir": DOCUMENTATION_DIR,
    "example_dir": EXAMPLE_DIR
}

# Ensure all directories exist
for path_name, path_value in PATHS.items():
    if path_value and not os.path.exists(path_value):
        os.makedirs(path_value, exist_ok=True)

# Configuration validation
VALID_SETTINGS = {
    "DEBUG_MODE": [True, False],
    "PRIVACY_COMPLIANCE": ["GDPR", "CCPA", "None"],
    "LOG_LEVEL": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "MAX_RETRY_ATTEMPTS": range(1, 11),
    "TIMEOUT_SECONDS": range(1, 301),
    "WEBCAM_INDEX": range(0, 10),
    "BATCH_SIZE": range(1, 65),
    "PROCESSING_THREADS": range(1, 17),
}

# Validate configuration
for setting, valid_values in VALID_SETTINGS.items():
    if setting in globals():
        value = globals()[setting]
        if isinstance(valid_values, range):
            if value not in valid_values:
                raise ValueError(f"Invalid value for {setting}: {value}")
        elif value not in valid_values:
            raise ValueError(f"Invalid value for {setting}: {value}")

print("Configuration loaded successfully")