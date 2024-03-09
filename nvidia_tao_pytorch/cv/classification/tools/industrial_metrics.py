# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Module for computing industrial metrics """

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score


def log(msg, log_file):
    """
    Print a message and write it to the log file.

    Parameters:
        msg (str): The message to be printed and logged.
    """
    print(msg)
    log_file.write(msg + '\n')


def calculate_metrics(csv_path, num_classes, no_defect_class, threshold):
    """
    Calculate and log various metrics for a defect classification model.

    Parameters:
        csv_path (str): Path to the CSV file containing evaluation results.
        num_classes (int): Number of classes in the dataset.
        no_defect_class (str): Name of the no-defect class.
        threshold (float): Threshold value for binary classification.
    """
    # Create a log file path
    log_file_path = os.path.join(os.path.dirname(csv_path), 'industrial_metrics.txt')

    # Open the log file for writing
    with open(log_file_path, 'w') as log_file:

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Get the column names as label order
        labels = df.iloc[:, 1:num_classes + 1].columns.tolist()
        label_map = {label: index for index, label in enumerate(sorted(labels))}

        # Extract relevant columns
        pred_scores = df.iloc[:, 1:num_classes + 1].values
        pred_score_no_defect = pred_scores[:, df.columns.get_loc(no_defect_class) - 1]  # Get index based on column name
        true_labels_class_name = df["gt_label"].values  # Assuming the ground truth label column is named "gt_label"
        true_labels_binary = (true_labels_class_name != no_defect_class).astype(int)
        true_labels = df['gt_label'].map(label_map).values

        log("Multi-Class Metrics:", log_file)
        # Original Confusion Matrix (multi-class)
        predictions_multiclass = np.argmax(pred_scores, axis=1)
        log("\nConfusion Matrix (Multi-Class):", log_file)
        log(str(pd.crosstab(true_labels, predictions_multiclass, rownames=['True'], colnames=['Predicted'], margins=True)), log_file)

        OK_indexes = [i for i in range(0, len(true_labels)) if true_labels[i] == label_map['OK']]
        OK_predicted = np.take(predictions_multiclass, OK_indexes)
        OverKill_indexes = [i for i in range(0, len(OK_predicted)) if OK_predicted[i] != label_map['OK']]
        log(f'\nOverkill count: {len(OverKill_indexes)}, rate: {len(OverKill_indexes)/len(OK_indexes)}', log_file)

        NG_indexes = [i for i in range(0, len(true_labels)) if true_labels[i] != label_map['OK']]
        NG_predicted = np.take(predictions_multiclass, NG_indexes)
        UnderKill_indexes = [i for i in range(0, len(NG_predicted)) if NG_predicted[i] == label_map['OK']]
        log(f'\nUnderkill count: {len(UnderKill_indexes)}, rate: {len(UnderKill_indexes)/len(NG_indexes)}', log_file)

        # Overall Accuracy
        overall_accuracy = accuracy_score(true_labels, np.argmax(pred_scores, axis=1)) * 100
        log(f"\nOverall Accuracy: {overall_accuracy:.4f}%", log_file)
        log("\n--------------------------------------------------", log_file)

        log("\nBinary Classification Metrics:", log_file)
        # Binary Classification using the score of no_defect_class
        binary_predictions = (pred_score_no_defect < threshold).astype(int)

        # Binary Confusion Matrix
        log("\nConfusion Matrix (Binary): ", log_file)
        log(str(pd.crosstab(true_labels_binary, binary_predictions, rownames=['True'], colnames=['Predicted'], margins=True)), log_file)

        # Binary Defect Accuracy
        binary_defect_accuracy = accuracy_score(true_labels_binary, binary_predictions) * 100
        log(f"\nBinary Accuracy: {binary_defect_accuracy:.4f}%", log_file)

        # Recall for Defective Class (Class 1)
        recall_defective = recall_score(true_labels_binary, binary_predictions, pos_label=1) * 100
        log(f"\nRecall for Defective Class (Defect Accuracy): {recall_defective:.4f}%", log_file)
        # False Alarm Rate (False Positive Rate)
        false_positive_rate = np.sum((true_labels_binary == 0) & (binary_predictions == 1)) / len(true_labels_binary) * 100
        log(f"False Alarm Rate (False Positive Rate): {false_positive_rate:.4f}%", log_file)


if __name__ == "__main__":
    """main function."""
    parser = argparse.ArgumentParser(description="Evaluate metrics for a defect classification model.")
    parser.add_argument("-i", "--input_csv_path", help="Path to the CSV file containing evaluation results", required=True)
    parser.add_argument("-nc", "--num_classes", type=int, help="Number of classes in the dataset", required=True)
    parser.add_argument("-nd", "--no_defect_class", help="Name of the no-defect class", required=True)
    parser.add_argument("-t", "--threshold", type=float, help="Threshold value for binary classification", required=True)

    args = parser.parse_args()

    # Calculate industrial metrics
    calculate_metrics(args.input_csv_path, args.num_classes, args.no_defect_class, args.threshold)
