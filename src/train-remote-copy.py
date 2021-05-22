import os
import argparse
from typing import no_type_check
import pandas as pd
import numpy as  np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score




from model import model

from azureml.core import Run

run = Run.get_context()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='data',
        help='Path to the training data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for SGD'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD'
    )

    args = parser.parse_args()

    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    #preparar datos
    #df = pd.read_csv('./data/data.csv', delimiter=';')
    df = pd.read_csv(args.data_path, delimiter=';')
    print(df.head(5))


