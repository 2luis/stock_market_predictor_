"""
Centralized logging configuration module.
This script provides a consistent logging setup across the project with file and console output.
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict

class Logger:
    """
    centralized logging configuration class.
    maintains singleton loggers for different modules.
    """
    
    # class variables to store logger instances
    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False
    
    @classmethod
    def setup(cls, 
              name: str = __name__,
              log_dir: str = 'logs',
              level: int = logging.INFO) -> logging.Logger:
        """
        setup logging configuration if not already initialized.
        
        args:
            name (str): name of the logger, typically __name__
            log_dir (str): directory to store log files
            level (int): logging level (e.g., logging.INFO)
            
        returns:
            logging.Logger: configured logger instance
        """
        if not cls._initialized:
            # create log directory
            os.makedirs(log_dir, exist_ok=True)
            
            # configure logging
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
        """
        get existing logger or create new one for the specific module.
        
        args:
            name (str): name of the logger to retrieve
            
        returns:
            logging.Logger: logger instance for the specified name
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]