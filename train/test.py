import os
import sys
# Add current directory and parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())
from models.VLV_stage1 import test


def main():
    print("Hello, World!")
    test()

if __name__ == "__main__":
    main()