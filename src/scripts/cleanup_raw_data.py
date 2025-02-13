#!/usr/bin/env python3
"""
Cleanup Script

This script reads the configuration for raw data files and deletes them after analysis is complete.
"""

import os
import json

from utils.logger import get_logger

logger = get_logger('cleanup_raw_data')


def cleanup_raw_data():
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config.json: {str(e)}")
        return
        
    files_to_delete = config.get('files', {})
    
    for key, file_path in files_to_delete.items():
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted raw data file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
        else:
            logger.info(f"File {file_path} does not exist.")


if __name__ == "__main__":
    cleanup_raw_data() 