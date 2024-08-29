import argparse
from data.preprocessing.data import CholecSeg8k


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CholecSeg8K Dataset Preprocessing")
    parser.add_argument("--data_root", default="datasets/cholecseg8k", required=True, type=str, help="Path to cholecSeg8k dataset")
    parser.add_argument("--anno_mode", default="tissue", required=True, type=str, help="Annotation mode: all or tissue (four first classes)")
    args = parser.parse_args()

    cholecseg8k = CholecSeg8k(args.data_root, args.anno_mode)
    cholecseg8k.load_data()
    cholecseg8k.preprocess()
