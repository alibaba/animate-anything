"""Shared pytest fixtures and configuration for all tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> DictConfig:
    """Create a mock configuration for testing."""
    config = OmegaConf.create({
        "model": {
            "name": "test_model",
            "batch_size": 1,
            "learning_rate": 0.001,
        },
        "data": {
            "train_path": "/tmp/train",
            "val_path": "/tmp/val",
            "num_workers": 2,
        },
        "training": {
            "epochs": 10,
            "save_interval": 100,
            "log_interval": 50,
        },
    })
    return config


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample image file path for testing."""
    image_path = temp_dir / "test_image.jpg"
    # Create a dummy file to simulate an image
    image_path.write_text("dummy image content")
    return image_path


@pytest.fixture
def sample_video_path(temp_dir: Path) -> Path:
    """Create a sample video file path for testing."""
    video_path = temp_dir / "test_video.mp4"
    # Create a dummy file to simulate a video
    video_path.write_text("dummy video content")
    return video_path


@pytest.fixture
def mock_model_weights(temp_dir: Path) -> Path:
    """Create mock model weights file for testing."""
    weights_path = temp_dir / "model_weights.pth"
    weights_path.write_text("dummy weights")
    return weights_path


@pytest.fixture
def environment_setup(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    monkeypatch.setenv("TEST_MODE", "1")
    yield
    # Environment variables are automatically restored by monkeypatch


@pytest.fixture
def mock_dataset_config() -> dict:
    """Create a mock dataset configuration."""
    return {
        "dataset_name": "test_dataset",
        "image_size": 256,
        "video_length": 16,
        "fps": 8,
        "augmentation": {
            "random_flip": True,
            "color_jitter": 0.1,
        },
    }


@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Clean up CUDA cache after each test to prevent memory issues."""
    yield
    # Only import torch if it's available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@pytest.fixture
def mock_training_args() -> dict:
    """Create mock training arguments."""
    return {
        "output_dir": "/tmp/test_output",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": "/tmp/logs",
        "save_steps": 1000,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
    }


# Markers for slow tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )