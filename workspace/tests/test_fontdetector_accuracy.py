import pytest
import numpy as np

# Overall accuracy must be greater than 85%
def test_overall_accuracy(predictions):
    all_labels, all_predictions = predictions
    acc = (all_predictions == all_labels).sum() / len(all_labels) * 100
    assert acc > 85, f"Overall accuracy too low: {acc:.2f}%"


# Per-class accuracy must be greater than 75% for every class
def test_per_class_accuracy(predictions, font_list):
    all_labels, all_predictions = predictions
    num_classes = len(font_list)
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    failed_classes = []

    for true_label, pred_label in zip(all_labels, all_predictions):
        total[true_label] += 1
        if true_label == pred_label:
            correct[true_label] += 1

    for cls in range(num_classes):
        if total[cls] == 0:
            continue  # Skip classes with no examples
        acc = correct[cls] / total[cls] * 100
        if acc < 75:
            failed_classes.append((font_list[cls], acc))

    if failed_classes:
        summary = ", ".join([f"{cls} ({acc:.2f}%)" for cls, acc in failed_classes])
        pytest.fail(f"Some font classes have low accuracy: {summary}")
