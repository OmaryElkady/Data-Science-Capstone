"""Vector-index bookkeeping for the assembled feature vector.

A one-hot column occupies one slot per category, so a column's position among the
VectorAssembler inputs is not its index in the vector, and the vector's `ml_attr` metadata
does not survive StandardScaler plus a Delta round-trip. The expansion is computed here,
written to the feature manifest by 04_gold, and read by 05_train and 07_score.

Imports PySpark (it inspects a fitted PipelineModel), so CI does not unit-test it.
"""

from __future__ import annotations

from pyspark.ml.feature import OneHotEncoderModel, VectorAssembler


def _encoder_width(encoder: OneHotEncoderModel, output_col: str) -> int:
    """Number of vector slots this encoder contributes for `output_col`."""
    try:
        out_cols = list(encoder.getOutputCols())
    except (KeyError, TypeError):
        # An encoder built with the singular inputCol/outputCol setters raises
        # rather than returning a list. Observed on Databricks serverless.
        out_cols = [encoder.getOutputCol()]
    size = encoder.categorySizes[out_cols.index(output_col)]
    return size - 1 if encoder.getDropLast() else size


def expand_feature_space(pipeline_model) -> list[dict]:
    """Map every vector slot to the column that produced it.

    Returns one dict per slot: `vector_index`, `name`, `source_column`,
    `attr_type`. `name` is unique and is what `05_train` looks `dep_delay` up by;
    `source_column` is the pre-encoding column, so all slots of a one-hot block
    share it.
    """
    assembler = next(s for s in pipeline_model.stages if isinstance(s, VectorAssembler))
    encoders = {
        out: stage
        for stage in pipeline_model.stages
        if isinstance(stage, OneHotEncoderModel)
        for out in (list(stage.getOutputCols())
                    if _has_plural_outputs(stage) else [stage.getOutputCol()])
    }

    rows, cursor = [], 0
    for column in assembler.getInputCols():
        if column in encoders:
            width = _encoder_width(encoders[column], column)
            for offset in range(width):
                rows.append({
                    "vector_index": cursor,
                    "name": f"{column}[{offset}]",
                    "source_column": column,
                    "attr_type": "one_hot",
                })
                cursor += 1
        else:
            rows.append({
                "vector_index": cursor,
                "name": column,
                "source_column": column,
                "attr_type": "numeric_or_boolean",
            })
            cursor += 1
    return rows


def _has_plural_outputs(stage) -> bool:
    try:
        list(stage.getOutputCols())
        return True
    except (KeyError, TypeError):
        return False


def index_by_name(manifest_rows) -> dict[str, int]:
    """`{name: vector_index}` from manifest rows (Spark Rows or dicts)."""
    out = {}
    for row in manifest_rows:
        item = row.asDict() if hasattr(row, "asDict") else row
        out[item["name"]] = int(item["vector_index"])
    return out
