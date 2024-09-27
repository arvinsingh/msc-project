import os
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """
    Describes the common system setting needed for reproducible training
    """

    seed: int = 21  # Seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # Enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # Make cudnn deterministic (reproducible training)


@dataclass
class TrainingConfig:
    """
    Describes configuration of the training process
    """

    num_classes: int = 2
    batch_size: int = 32
    epochs_count: int = 20
    init_learning_rate: float = 0.001  # Initial learning rate
    weight_decay: float = 0.00001
    data_root: str = "..\\data\\processed\\"  # Root directory path of the dataset
    num_workers: int = 2
    device: str = "cuda"

    # For tensorboard logging and saving checkpoints
    save_audio_model_name: str = "fluency_classifier.pt"
    save_landmark_model_name: str = "landmark_classifier.pt"
    save_combined_model_name: str = "combined_classifier.pt"
    root_log_dir: str = os.path.join("..\\output\\Logs_Checkpoints", "Model_logs")
    root_checkpoint_dir: str = os.path.join("..\\output\\Logs_Checkpoints", "Model_checkpoints")

    # Threshold for distance comparison
    threshold: float = 0.3

    # Current log and checkpoint directory.
    log_dir: str = "version_0"
    checkpoint_dir: str = "version_0"