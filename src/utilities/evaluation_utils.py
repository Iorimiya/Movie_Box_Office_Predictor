from sklearn.metrics import classification_report, accuracy_score


def classification_report_to_string(true_labels: list[int], predicted_labels: list[int], target_names=None) -> None:
    if target_names is None:
        target_names = ['Negative (0)', 'Positive (1)']
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))
