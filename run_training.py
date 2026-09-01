import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train

if __name__ == "__main__":
    train()