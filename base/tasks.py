import os
import subprocess
import logging
from celery import shared_task

logging.basicConfig(level=logging.INFO)

@shared_task
def run_data_script():
    script_path = os.path.join('scripts', 'data.py')
    logging.info(f"Running data script: {script_path}")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    logging.info(f"Data script result: {result.stdout}")
    if result.returncode == 0:
        return result.stdout
    else:
        logging.error(f"Data script error: {result.stderr}")
        raise ValueError(f"Error running data script: {result.stderr}")

@shared_task
def run_retrain_script():
    script_path = os.path.join('scripts', 'Retrain.py')
    logging.info(f"Running retrain script: {script_path}")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    logging.info(f"Retrain script result: {result.stdout}")
    if result.returncode == 0:
        return result.stdout
    else:
        logging.error(f"Retrain script error: {result.stderr}")
        raise ValueError(f"Error running retrain script: {result.stderr}")
