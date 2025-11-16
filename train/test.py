import os
import sys
# Add current directory and parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())
# from models.VLV_stage1 import test
from data import PointDataset


def main():
    print("Hello, World!")
    dataset = PointDataset(data_path='./data/modelnet40_normal_resampled', npoints=8192, subset='test')
    print(f"Dataset size: {len(dataset)}")
    print(dataset.datapath[0])  # Print the first data point
    print(dataset[0][2][0].shape)
    print(dataset.list_of_labels[0])  # Print the label of the first data point
    # test()

if __name__ == "__main__":
    main()