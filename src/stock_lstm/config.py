from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainConfig:
    symbol: str = "PETR4.SA"
    start_date: str = "2018-01-01"
    window_size: int = 30
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 20
    output_dir: str = "models/lstm_petr4"
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def artifact_dir(self) -> Path:
        return Path(self.output_dir)
