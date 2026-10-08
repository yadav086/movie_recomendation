import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config():
    # 1. Get the absolute path of the directory where THIS script is running (the 'scr' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Point directly to config.yaml inside that same 'scr' folder
    absolute_path = os.path.join(current_dir, 'config.yaml')
    
    try:
        with open(absolute_path, 'r') as read:
            return yaml.safe_load(read)
    except Exception as e:
        logger.exception(f'failed in load_config at path: {absolute_path}')
        raise
