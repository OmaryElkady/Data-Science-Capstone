"""Evaluation helpers for imbalanced binary classification on Spark.

Two constraints shaped this module:

1. Databricks serverless has no RDD API, so `pyspark.mllib`'s
   `BinaryClassificationMetrics` is unavailable. Everything here is
   DataFrame-only.
2. Spark's `BinaryClassificationEvaluator` returns ROC-AUC and PR-AUC but
   no threshold-dependent metrics, and `MulticlassClassificationEvaluator`
   defaults to weighted averages that flatter the majority class. The
   positive class is the one we care about, so metrics here are computed
   for label 1 explicitly.

Every sweep is a single pass over the data — thresholds are evaluated as
parallel aggregations rather than in a Python loop of `.count()` calls.
"""

from __future__ import annotations

from typing import Iterable

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

POSITIVE_PROBABILITY = "p1"


def with_positive_probability(
    df: DataFrame, probability_col: str = "probability", out_col: str = POSITIVE_PROBABILITY
) -> DataFrame:
    """Extract P(label=1) from Spark's probability vector into a double column."""
    return df.withColumn(out_col, vector_to_array(F.col(probability_col))[1])


def area_under(df: DataFrame, metric: str = "areaUnderROC", label_col: str = "label") -> float:
    """ROC-AUC or PR-AUC. Threshold-independent — measures ranking, not decisions."""
    return BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="probability", metricName=metric
    ).evaluate(df)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def confusion_at(
    df: DataFrame,
    thresholds: Iterable[float],
    prob_col: str = POSITIVE_PROBABILITY,
    label_col: str = "label",
) -> list[dict]:
    """TP/FP/FN/TN for every threshold in one pass over the data.

    Returns a list of dicts, one per threshold, with precision, recall,
    F1 and the counts behind them.
    """
    thresholds = list(thresholds)
    y = F.col(label_col).cast("int")

    aggregations = [F.count("*").alias("n"), F.sum(y).alias("positives")]
    for i, t in enumerate(thresholds):
        predicted = (F.col(prob_col) >= F.lit(float(t))).cast("int")
        aggregations += [
            F.sum(predicted * y).alias(f"tp{i}"),
            F.sum(predicted * (1 - y)).alias(f"fp{i}"),
            F.sum((1 - predicted) * y).alias(f"fn{i}"),
        ]

    row = df.agg(*aggregations).first().asDict()
    n, positives = row["n"], row["positives"]

    results = []
    for i, t in enumerate(thresholds):
        tp, fp, fn = row[f"tp{i}"], row[f"fp{i}"], row[f"fn{i}"]
        tn = n - tp - fp - fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        results.append(
            {
                "threshold": float(t),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": _safe_div(2 * precision * recall, precision + recall),
                "accuracy": _safe_div(tp + tn, n),
                "predicted_positive_rate": _safe_div(tp + fp, n),
                "base_rate": _safe_div(positives, n),
            }
        )
    return results


def fbeta(precision: float, recall: float, beta: float = 1.0) -> float:
    """F-beta. beta > 1 weights recall above precision.

    For flight delay, a missed delay (false negative) costs a connection
    while a false alarm costs a needless rebooking. If that asymmetry is
    quantified, beta should reflect it rather than defaulting to 1.
    """
    b2 = beta * beta
    return _safe_div((1 + b2) * precision * recall, b2 * precision + recall)


def best_threshold(sweep: list[dict], beta: float = 1.0) -> dict:
    """Threshold maximising F-beta. Select on validation, never on test.

    Under class imbalance the optimal threshold is essentially never 0.5,
    so leaving Spark's default in place silently costs recall.
    """
    scored = [dict(row, fbeta=fbeta(row["precision"], row["recall"], beta)) for row in sweep]
    return max(scored, key=lambda r: r["fbeta"])


def brier_score(
    df: DataFrame, prob_col: str = POSITIVE_PROBABILITY, label_col: str = "label"
) -> float:
    """Mean squared error of the predicted probability.

    Decomposes into calibration + refinement, so it penalises a model that
    ranks well but reports probabilities that do not mean what they say.
    Matters here because the app surfaces the probability to a user.
    """
    return df.select(
        F.avg(F.pow(F.col(prob_col) - F.col(label_col).cast("double"), 2))
    ).first()[0]


def calibration_bins(
    df: DataFrame,
    n_bins: int = 10,
    prob_col: str = POSITIVE_PROBABILITY,
    label_col: str = "label",
):
    """Predicted vs observed frequency per probability bin (reliability diagram).

    A perfectly calibrated model sits on the diagonal: of the flights it
    called 30% likely to be delayed, 30% were delayed.
    """
    return (
        df.withColumn("bin", F.least(F.floor(F.col(prob_col) * n_bins), F.lit(n_bins - 1)))
        .groupBy("bin")
        .agg(
            F.avg(prob_col).alias("mean_predicted"),
            F.avg(F.col(label_col).cast("double")).alias("observed_rate"),
            F.count("*").alias("n"),
        )
        .orderBy("bin")
        .toPandas()
    )


def full_report(
    df: DataFrame, threshold: float, label_col: str = "label", beta: float = 1.0
) -> dict:
    """Every headline metric at one decision threshold, plus the AUCs."""
    scored = with_positive_probability(df)
    row = confusion_at(scored, [threshold], label_col=label_col)[0]
    row["fbeta"] = fbeta(row["precision"], row["recall"], beta)
    row["roc_auc"] = area_under(df, "areaUnderROC", label_col)
    row["pr_auc"] = area_under(df, "areaUnderPR", label_col)
    row["brier"] = brier_score(scored, label_col=label_col)
    return row
