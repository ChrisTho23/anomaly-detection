import subprocess

if __name__ == "__main__":
    subprocess.run(['python', 'train.py'])
    subprocess.run(['python', 'prediction.py'])
    subprocess.run(['python', 'find_annomalies.py'])
    subprocess.run(['python', 'clean_up.py'])