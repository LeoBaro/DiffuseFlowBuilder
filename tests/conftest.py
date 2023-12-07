from genericpath import exists
import pytest
from pathlib import Path

@pytest.fixture
def root():
    return Path(__file__).parent.absolute()

@pytest.fixture
def data_dir(root):
    return root / "data"

@pytest.fixture
def output_dir(root):
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir