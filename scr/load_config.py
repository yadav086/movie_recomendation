import os
import yaml
import logging

logger = logging.getLogger(__name__)

def load_config():
    # 1. Get the absolute directory where this load_config.py script lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go one folder up (to the root directory) where config.yaml lives
    root_dir = os.path.dirname(current_dir)
    
    # 3. Create a reliable, absolute path to config.yaml
    absolute_path = os.path.join(root_dir, 'config.yaml')
    
    try:
        with open(absolute_path, 'r') as read:
            return yaml.safe_load(read)
    except Exception as e:
        logger.exception(f'failed in load_config at path: {absolute_path}')
        raise
