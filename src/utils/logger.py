import logging
import os
from datetime import datetime
from typing import Optional, Dict

class Logger:
    """Centralized logging configuration"""
    
    _loggers: Dict[str, logging.Logger] = {}  # Store multiple loggers
    _initialized: bool = False
    
    @classmethod
    def setup(cls, 
              name: str = __name__,
              log_dir: str = 'logs',
              level: int = logging.INFO) -> logging.Logger:
        """Setup logging configuration if not already done"""
        if not cls._initialized:
            # Create log directory
            os.makedirs(log_dir, exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(
                        f'{log_dir}/data_collection_{datetime.now().strftime("%Y%m%d")}.log'
                    ),
                    logging.StreamHandler()
                ]
            )
            cls._initialized = True
            
        return cls.get_logger(name)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get existing logger or create new one for the specific module"""
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]