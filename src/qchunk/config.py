from dataclasses import dataclass

from qchunk.agent import QAggregation


@dataclass(frozen=True)
class QCConfig:
    """Hyperparameters required by the core QC learner."""

    horizon: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    num_critics: int = 2
    num_candidates: int = 32
    num_flow_steps: int = 10
    discount: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    q_aggregation: QAggregation = "mean"
    flow_layer_norm: bool = False
    critic_layer_norm: bool = True

    @property
    def action_chunk_dim(self) -> int:
        return self.horizon * self.action_dim

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.num_critics <= 0:
            raise ValueError("num_critics must be positive")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.num_flow_steps <= 0:
            raise ValueError("num_flow_steps must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must lie in [0, 1]")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.q_aggregation not in ("mean", "min"):
            raise ValueError("q_aggregation must be 'mean' or 'min'")
