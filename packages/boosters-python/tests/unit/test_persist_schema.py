"""Tests for the persistence schema module.

These tests validate that pydantic models can parse JSON produced by Rust
and that the parsed data matches expected structures.
"""

import json

import numpy as np
import pytest

from boosters import Dataset, GBDTConfig, GBDTModel, GBLinearConfig, GBLinearModel, Objective
from boosters.persist.schema import (
    GBDTModelSchema,
    GBLinearModelSchema,
    JsonEnvelope,
)


def make_regression_data(n_samples: int = 200, n_features: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n_samples) * 0.1).astype(np.float32)
    return X, y


class TestJsonEnvelopeParsing:
    """Test JSON envelope parsing from Rust-generated JSON."""

    @pytest.fixture
    def gbdt_model(self) -> GBDTModel:
        """Create a simple trained GBDT model."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(200) * 0.1).astype(np.float32)

        return GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=10))

    @pytest.fixture
    def gblinear_model(self) -> GBLinearModel:
        """Create a simple trained GBLinear model."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        return GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=50))

    def test_gbdt_json_parses_with_pydantic(self, gbdt_model: GBDTModel) -> None:
        """GBDT JSON from Rust parses successfully with pydantic."""
        json_bytes = gbdt_model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        # Parse with pydantic
        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)

        # Verify envelope structure
        assert envelope.bstr_version == 2
        assert envelope.model_type == "gbdt"
        assert isinstance(envelope.model, GBDTModelSchema)

    def test_gblinear_json_parses_with_pydantic(self, gblinear_model: GBLinearModel) -> None:
        """GBLinear JSON from Rust parses successfully with pydantic."""
        json_bytes = gblinear_model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        # Parse with pydantic
        envelope = JsonEnvelope[GBLinearModelSchema].model_validate_json(json_str)

        # Verify envelope structure
        assert envelope.bstr_version == 2
        assert envelope.model_type == "gblinear"
        assert isinstance(envelope.model, GBLinearModelSchema)

    def test_gbdt_model_structure_matches_rust(self, gbdt_model: GBDTModel) -> None:
        """Parsed GBDT model structure matches expected values from Rust."""
        json_bytes = gbdt_model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)
        model = envelope.model

        # Check metadata
        assert model.meta.num_features == 5
        assert model.meta.num_groups == 1

        # Check forest
        assert len(model.forest.trees) == 10  # n_estimators=10
        assert model.forest.n_groups == 1
        assert len(model.forest.base_score) == 1

        # Output transform is persisted for correct post-processing
        assert model.output_transform == "identity"

        # Check trees have valid structure
        for tree in model.forest.trees:
            assert tree.num_nodes >= 1
            # Internal nodes have split info
            n_internals = len(tree.split_indices)
            assert len(tree.thresholds) == n_internals
            assert len(tree.children_left) == n_internals
            assert len(tree.children_right) == n_internals
            assert len(tree.default_left) == n_internals

    def test_gblinear_model_structure_matches_rust(self, gblinear_model: GBLinearModel) -> None:
        """Parsed GBLinear model structure matches expected values from Rust."""
        json_bytes = gblinear_model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        envelope = JsonEnvelope[GBLinearModelSchema].model_validate_json(json_str)
        model = envelope.model

        # Check metadata
        assert model.meta.num_features == 5
        assert model.meta.num_groups == 1

        # Check weights
        assert model.weights.num_features == 5
        assert model.weights.num_groups == 1
        # weights include bias term: (num_features + 1) * num_groups
        assert len(model.weights.values) == 6  # 5 features + 1 bias

        # Base score may be empty for models trained by booste-rs (it can be baked into the bias term)
        assert len(model.base_score) in {0, 1}

        # Output transform is persisted for correct post-processing
        assert model.output_transform == "identity"

        # GBLinear stores base score in the bias term of the linear model,
        # so schema.base_score is empty (or can be non-empty if explicitly set)
        # This is an implementation detail - we just verify the schema parses

    def test_gbdt_objective_preserved(self, gbdt_model: GBDTModel) -> None:
        """Output transform is preserved in serialized model."""
        json_bytes = gbdt_model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)

        assert envelope.model.output_transform == "identity"

    def test_gblinear_objective_preserved(self, gblinear_model: GBLinearModel) -> None:
        """Output transform is preserved in serialized model."""
        json_bytes = gblinear_model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        envelope = JsonEnvelope[GBLinearModelSchema].model_validate_json(json_str)

        assert envelope.model.output_transform == "identity"


class TestBinaryClassification:
    """Test schema with binary classification models."""

    def test_gbdt_binary_parses(self) -> None:
        """Binary classification GBDT model parses correctly."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=10, objective=Objective.logistic()))

        json_str = model.to_json_bytes().decode("utf-8")
        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)

        assert envelope.model.meta.num_groups == 1
        assert envelope.model.output_transform == "sigmoid"


class TestMulticlass:
    """Test schema with multiclass classification models."""

    def test_gbdt_multiclass_parses(self) -> None:
        """Multiclass GBDT model parses correctly."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 5)).astype(np.float32)
        y = (np.digitize(X[:, 0], bins=[-0.5, 0.5])).astype(np.float32)  # 3 classes

        model = GBDTModel.train(
            Dataset(X, y), config=GBDTConfig(n_estimators=10, objective=Objective.softmax(n_classes=3))
        )

        json_str = model.to_json_bytes().decode("utf-8")
        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)

        assert envelope.model.meta.num_groups == 3
        assert envelope.model.output_transform == "softmax"


class TestSchemaValidation:
    """Test pydantic validation catches invalid data."""

    def test_invalid_task_kind_rejected(self) -> None:
        """Invalid task kind is rejected by pydantic."""
        invalid_json = json.dumps({
            "bstr_version": 2,
            "model_type": "gbdt",
            "model": {
                "meta": {
                    "task": "invalid_task",  # Invalid
                    "num_features": 5,
                },
                "forest": {
                    "trees": [],
                    "n_groups": 1,
                    "base_score": [0.0],
                },
                "output_transform": "identity",
            },
        })

        with pytest.raises(Exception):  # pydantic validation error  # noqa: B017, PT011
            JsonEnvelope[GBDTModelSchema].model_validate_json(invalid_json)

    def test_missing_required_field_rejected(self) -> None:
        """Missing required field is rejected by pydantic."""
        invalid_json = json.dumps({
            "bstr_version": 2,
            "model_type": "gbdt",
            "model": {
                "meta": {
                    # Missing 'task' and 'num_features'
                },
                "forest": {},
                "output_transform": "identity",
            },
        })

        with pytest.raises(Exception):  # pydantic validation error  # noqa: B017, PT011
            JsonEnvelope[GBDTModelSchema].model_validate_json(invalid_json)


class TestCrossLanguageRoundTrip:
    """Test cross-language JSON round-trip: Rust → Python → Rust."""

    def test_gbdt_roundtrip_rust_python_rust(self) -> None:
        """GBDT: Rust JSON → Python pydantic → JSON → Rust loads correctly."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=10))

        # Step 1: Rust → JSON
        original_preds = model.predict(Dataset(X))
        json_bytes = model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        # Step 2: JSON → Python pydantic
        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)

        # Step 3: Python pydantic → JSON
        roundtrip_json = envelope.model_dump_json(by_alias=True, exclude_none=True)

        # Step 4: JSON → Rust (via from_json_bytes)
        loaded = GBDTModel.from_json_bytes(roundtrip_json.encode("utf-8"))
        loaded_preds = loaded.predict(Dataset(X))

        # Verify predictions match
        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_gblinear_roundtrip_rust_python_rust(self) -> None:
        """GBLinear: Rust JSON → Python pydantic → JSON → Rust loads correctly."""
        X, y = make_regression_data()
        model = GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=50))

        # Step 1: Rust → JSON
        original_preds = model.predict(Dataset(X))
        json_bytes = model.to_json_bytes()
        json_str = json_bytes.decode("utf-8")

        # Step 2: JSON → Python pydantic
        envelope = JsonEnvelope[GBLinearModelSchema].model_validate_json(json_str)

        # Step 3: Python pydantic → JSON
        roundtrip_json = envelope.model_dump_json(by_alias=True, exclude_none=True)

        # Step 4: JSON → Rust (via from_json_bytes)
        loaded = GBLinearModel.from_json_bytes(roundtrip_json.encode("utf-8"))
        loaded_preds = loaded.predict(Dataset(X))

        # Verify predictions match
        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_binary_classification_roundtrip(self) -> None:
        """Binary classification model survives round-trip."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=10, objective=Objective.logistic()))

        original_preds = model.predict(Dataset(X))
        json_str = model.to_json_bytes().decode("utf-8")

        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)
        roundtrip_json = envelope.model_dump_json(by_alias=True, exclude_none=True)

        loaded = GBDTModel.from_json_bytes(roundtrip_json.encode("utf-8"))
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_multiclass_roundtrip(self) -> None:
        """Multiclass model survives round-trip."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 5)).astype(np.float32)
        y = (np.digitize(X[:, 0], bins=[-0.5, 0.5])).astype(np.float32)

        model = GBDTModel.train(
            Dataset(X, y), config=GBDTConfig(n_estimators=10, objective=Objective.softmax(n_classes=3))
        )

        original_preds = model.predict(Dataset(X))
        json_str = model.to_json_bytes().decode("utf-8")

        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_str)
        roundtrip_json = envelope.model_dump_json(by_alias=True, exclude_none=True)

        loaded = GBDTModel.from_json_bytes(roundtrip_json.encode("utf-8"))
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)
