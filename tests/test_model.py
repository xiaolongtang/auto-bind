"""Shape, normalization, sharing, loss, and save/load tests for the model."""

import torch

from metadata_matcher.dataset import LABEL_MAP
from metadata_matcher.losses import cosine_embedding_loss, cosine_targets_from_labels
from metadata_matcher.model import FieldEncoder, PairClassifier, SiameseFieldMatcher


def test_field_encoder_shape_and_l2_norm() -> None:
    torch.manual_seed(42)
    encoder = FieldEncoder(vocab_size=32)
    token_ids = torch.randint(0, 32, (7, 64), dtype=torch.long)

    embeddings = encoder(token_ids)

    assert embeddings.shape == (7, 128)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(7),
        rtol=1e-5,
        atol=1e-6,
    )


def test_field_encoder_handles_sequence_shorter_than_largest_kernel() -> None:
    encoder = FieldEncoder(vocab_size=8, kernel_sizes=(2, 3, 4, 5))
    embeddings = encoder(torch.tensor([[2], [3]], dtype=torch.long))
    assert embeddings.shape == (2, 128)


def test_pair_classifier_output_shape_and_feature_width() -> None:
    classifier = PairClassifier(embedding_dim=128)
    embedding_a = torch.randn(5, 128)
    embedding_b = torch.randn(5, 128)

    features = classifier.pair_features(embedding_a, embedding_b)
    logits = classifier(embedding_a, embedding_b)

    assert features.shape == (5, 512)
    assert logits.shape == (5, 3)


def test_siamese_matcher_uses_one_shared_encoder() -> None:
    encoder = FieldEncoder(vocab_size=16)
    matcher = SiameseFieldMatcher(encoder, PairClassifier())
    a_ids = torch.randint(0, 16, (3, 12))
    b_ids = torch.randint(0, 16, (3, 12))

    assert matcher.encoder is encoder
    assert matcher(a_ids, b_ids).shape == (3, 3)
    assert sum(1 for module in matcher.modules() if isinstance(module, FieldEncoder)) == 1


def test_cosine_targets_and_loss_are_finite() -> None:
    labels = torch.tensor(
        [LABEL_MAP["NO_MATCH"], LABEL_MAP["DIRECT"], LABEL_MAP["DERIVATION"]]
    )
    targets = cosine_targets_from_labels(labels)
    assert targets.tolist() == [-1.0, 1.0, 1.0]

    embedding_a = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    embedding_b = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    loss = cosine_embedding_loss(embedding_a, embedding_b, labels, margin=0.2)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_state_dict_save_load_preserves_predictions(tmp_path) -> None:
    torch.manual_seed(123)
    encoder = FieldEncoder(vocab_size=20)
    classifier = PairClassifier(dropout=0.2)
    encoder.eval()
    classifier.eval()
    a_ids = torch.randint(0, 20, (4, 16))
    b_ids = torch.randint(0, 20, (4, 16))

    with torch.inference_mode():
        expected = classifier(encoder(a_ids), encoder(b_ids))
    torch.save(encoder.state_dict(), tmp_path / "encoder.pt")
    torch.save(classifier.state_dict(), tmp_path / "classifier.pt")

    restored_encoder = FieldEncoder(vocab_size=20)
    restored_classifier = PairClassifier(dropout=0.2)
    restored_encoder.load_state_dict(
        torch.load(tmp_path / "encoder.pt", map_location="cpu", weights_only=True)
    )
    restored_classifier.load_state_dict(
        torch.load(tmp_path / "classifier.pt", map_location="cpu", weights_only=True)
    )
    restored_encoder.eval()
    restored_classifier.eval()
    with torch.inference_mode():
        actual = restored_classifier(
            restored_encoder(a_ids), restored_encoder(b_ids)
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

