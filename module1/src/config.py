from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class ProjectConfig:
    name: str
    seed: int
    run_id_mode: str
    run_id: str


@dataclass
class DataConfig:
    input_csv: str
    target: str
    num_cols: List[str]
    cat_cols: List[str]
    test_size: float


@dataclass
class ModelParams:
    n_estimators: int
    max_depth: Optional[int] = None
    seed: int | None = None

    def __init__(self, **kwargs):
        self.n_estimators = kwargs.pop("n_estimators", 300)
        self.max_depth = kwargs.pop("max_depth", None)
        self.extra = kwargs

    def to_dict(self) -> Dict[str, Any]:
        d = {"n_estimators": self.n_estimators, "max_depth": self.max_depth}
        d.update(self.extra)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ModelConfig:
    task: str
    name: str
    params: ModelParams


@dataclass
class OutputConfig:
    artifacts_dir: str
    logs_dir: str
    save_predictions: bool
    save_model: bool
    save_report: bool


@dataclass
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    output: OutputConfig

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            project=ProjectConfig(**raw["project"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(
                task=raw["model"]["task"],
                name=raw["model"]["name"],
                params=ModelParams(**raw["model"]["params"]),
            ),
            output=OutputConfig(**raw["output"]),
        )
