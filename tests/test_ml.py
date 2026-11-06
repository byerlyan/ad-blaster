import csv
import numpy as np

from detection_pipeline import Classification, DetectionInput
from ml import (
    FEATURE_NAMES,
    FeatureDatasetStore,
    FeatureDatasetWriter,
    MLCommercialDetector,
    RuntimeFeatureBuilder,
)
from scripts.train_commercial_model import train


def test_runtime_feature_builder_has_stable_schema_and_temporal_features():
    builder = RuntimeFeatureBuilder(history_size=4)
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    visual = {"edge_score": 0.2, "template_score": 0.5, "color_score": 0.1}

    first = builder.build(black, visual, None)
    second = builder.build(white, visual, None)

    assert tuple(first.keys()) == FEATURE_NAMES
    assert first["scene_cut"] == 0.0
    assert second["scene_cut"] == 1.0
    assert second["recent_scene_cut_rate"] == 0.5
    assert all(np.isfinite(value) for value in second.values())


def test_dataset_store_persists_grouped_samples_and_exports_csv(tmp_path):
    store = FeatureDatasetStore(str(tmp_path / "features.sqlite3"))
    features = {name: float(index) for index, name in enumerate(FEATURE_NAMES)}
    store.append(
        timestamp=1.0,
        label="commercial",
        session_id="session-a",
        source_group="espn",
        features=features,
    )
    destination = tmp_path / "features.csv"

    count = store.export_csv(str(destination))
    row = next(csv.DictReader(open(destination, newline="")))

    assert count == 1
    assert row["label"] == "1"
    assert row["session_id"] == "session-a"
    assert row["source_group"] == "espn"


def test_background_dataset_writer_flushes(tmp_path):
    store = FeatureDatasetStore(str(tmp_path / "features.sqlite3"))
    writer = FeatureDatasetWriter(store)
    writer.submit(
        timestamp=1.0,
        label="content",
        session_id="session-a",
        source_group="abc",
        features={name: 0.0 for name in FEATURE_NAMES},
    )
    writer.flush()
    writer.close()

    assert store.count() == 1


def test_missing_and_corrupt_models_fall_back_to_unknown(tmp_path):
    missing = MLCommercialDetector(str(tmp_path / "missing.joblib"))
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_text("not a model")
    corrupt = MLCommercialDetector(str(corrupt_path))

    missing_result = missing.evaluate(
        DetectionInput(timestamp=1.0, features={"ml_inference_enabled": True})
    )
    corrupt_result = corrupt.evaluate(
        DetectionInput(timestamp=1.0, features={"ml_inference_enabled": True})
    )

    assert missing_result.classification == Classification.UNKNOWN
    assert not missing_result.available
    assert corrupt_result.classification == Classification.UNKNOWN
    assert not corrupt_result.available


def test_feature_collection_works_without_a_model(tmp_path):
    detector = MLCommercialDetector(str(tmp_path / "missing.joblib"))
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)

    actual = detector.evaluate(
        DetectionInput(
            timestamp=1.0,
            frame=frame,
            features={
                "collect_ml_features": True,
                "visual": {"edge_score": 0.4, "template_score": 0.5},
            },
        )
    )

    assert actual.classification == Classification.UNKNOWN
    assert set(actual.diagnostics["feature_vector"]) == set(FEATURE_NAMES)


class DummyModel:
    def predict_proba(self, values):
        return np.array([[0.1, 0.9]], dtype=np.float32)


def test_loaded_model_probability_is_marked_calibrated(tmp_path):
    detector = MLCommercialDetector(str(tmp_path / "missing.joblib"))
    detector._model = DummyModel()
    detector._error = None
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)

    actual = detector.evaluate(
        DetectionInput(
            timestamp=1.0,
            frame=frame,
            features={"ml_inference_enabled": True, "visual": {}},
        )
    )

    assert actual.classification == Classification.COMMERCIAL
    assert actual.calibrated_probability == np.float32(0.9)
    assert actual.confidence_kind == "calibrated_probability"


def test_training_uses_disjoint_groups_and_serializes_metadata(tmp_path):
    dataset = tmp_path / "training.csv"
    with open(dataset, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "label", "session_id", "source_group", *FEATURE_NAMES],
        )
        writer.writeheader()
        rng = np.random.default_rng(42)
        for group_index in range(16):
            label = group_index % 2
            for sample_index in range(6):
                base = 1.5 if label else -1.5
                features = {
                    name: base + float(rng.normal(0, 0.15))
                    for name in FEATURE_NAMES
                }
                writer.writerow(
                    {
                        "timestamp": group_index * 10 + sample_index,
                        "label": label,
                        "session_id": f"session-{group_index}",
                        "source_group": f"source-{group_index % 4}",
                        **features,
                    }
                )
    model_path = tmp_path / "model.joblib"

    metadata = train(str(dataset), str(model_path), seed=7)
    detector = MLCommercialDetector(str(model_path))

    fit = set(metadata["fit_groups"])
    calibration = set(metadata["calibration_groups"])
    test = set(metadata["test_groups"])
    assert not (fit & calibration or fit & test or calibration & test)
    assert metadata["commercial_precision"] >= 0.9
    assert metadata["commercial_recall"] >= 0.9
    assert detector.status().loaded
    assert detector.status().version == metadata["model_version"]
