import numpy as np
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from src.data.dataset import LABEL_NAMES


def compute_ner_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Вычисляет F1, precision, recall для NER.

    Args:
        predictions: массив предсказанных меток [batch, seq_len].
        labels: массив истинных меток [batch, seq_len].
            Значение -100 игнорируется (специальные токены).

    Returns:
        Словарь с метриками.
    """
    pred_labels, true_labels = _convert_to_label_strings(predictions, labels)

    return {
        "f1": f1_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
    }


def get_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> str:
    """
    Возвращает полный отчёт по классам.

    Args:
        predictions: массив предсказанных меток.
        labels: массив истинных меток.

    Returns:
        Строка с отчётом seqeval.
    """
    pred_labels, true_labels = _convert_to_label_strings(predictions, labels)
    return classification_report(true_labels, pred_labels)


def _convert_to_label_strings(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Конвертирует числовые метки в строки, убирая -100.

    Args:
        predictions: [batch, seq_len] числовые предсказания.
        labels: [batch, seq_len] числовые метки.

    Returns:
        Два списка списков строковых меток.
    """
    pred_label_strings = []
    true_label_strings = []

    for pred_row, label_row in zip(predictions, labels):
        pred_sent = []
        true_sent = []

        for pred, label in zip(pred_row, label_row):
            if label == -100:
                continue
            pred_sent.append(LABEL_NAMES[pred])
            true_sent.append(LABEL_NAMES[label])

        pred_label_strings.append(pred_sent)
        true_label_strings.append(true_sent)

    return pred_label_strings, true_label_strings


def make_compute_metrics():
    def compute_metrics(eval_pred) -> dict[str, float]:
        import numpy as np

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return compute_ner_metrics(predictions, labels)

    return compute_metrics
