"""Script to run the entire pipeline.
"""
import subprocess

if __name__ == "__main__":
    subprocess.run(['python', 'train.py'])
    subprocess.run(['python', 'prediction.py'])
    subprocess.run(['python', 'find_anomalies.py'])
    subprocess.run(['python', 'clean_up.py'])