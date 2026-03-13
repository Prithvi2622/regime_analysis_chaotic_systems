import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str, time_col: str = 'time') -> pd.DataFrame:
    """
    Loads financial time series data from CSV or Parquet files.
    
    Args:
        file_path: Path to the data file.
        time_col: Name of the column containing timestamp or index.
        
    Returns:
        pd.DataFrame containing at least 'time', 'price', and 'returns'.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
    
    logger.info(f"Loading data from {path}")
    
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.parquet', '.pqt']:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .csv or .parquet")
        
    # Standardize column naming if necessary, expecting time, price.
    if time_col not in df.columns:
        logger.warning(f"Time column '{time_col}' not found. Using index as time.")
        df['time'] = df.index
        
    return df
