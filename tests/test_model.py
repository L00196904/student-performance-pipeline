import os

def test_model_exists():
    assert os.path.exists(
        "artifacts/model/model.pkl"
    )