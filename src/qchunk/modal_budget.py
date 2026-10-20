from dataclasses import dataclass

# Modal public list prices captured on 2026-10-20. Keep these explicit so a
# benchmark report states the assumptions used for its estimate.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
PRICE_SNAPSHOT_DATE = "2026-10-20"


@dataclass(frozen=True)
class ModalResources:
    gpu: str = "A10"
    cpu_cores: float = 4.0
    memory_gib: float = 32.0

    def validate(self) -> None:
        if self.gpu not in GPU_USD_PER_SECOND:
            choices = ", ".join(GPU_USD_PER_SECOND)
            raise ValueError(f"unsupported GPU {self.gpu!r}; choose one of: {choices}")
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if self.memory_gib <= 0:
            raise ValueError("memory_gib must be positive")

    @property
    def usd_per_second(self) -> float:
        self.validate()
        return (
            GPU_USD_PER_SECOND[self.gpu]
            + self.cpu_cores * CPU_CORE_USD_PER_SECOND
            + self.memory_gib * MEMORY_GIB_USD_PER_SECOND
        )

    @property
    def usd_per_hour(self) -> float:
        return self.usd_per_second * 3600.0


def estimate_cost(resources: ModalResources, duration_seconds: float) -> float:
    if duration_seconds < 0:
        raise ValueError("duration_seconds cannot be negative")
    return resources.usd_per_second * duration_seconds


def budget_timeout_seconds(
    resources: ModalResources,
    max_spend_usd: float,
    *,
    platform_max_seconds: int = 24 * 60 * 60,
) -> int:
    """Convert a spend ceiling into a Modal timeout, capped by the platform maximum."""
    if max_spend_usd <= 0:
        raise ValueError("max_spend_usd must be positive")
    if platform_max_seconds <= 0:
        raise ValueError("platform_max_seconds must be positive")
    affordable_seconds = int(max_spend_usd / resources.usd_per_second)
    return max(1, min(affordable_seconds, platform_max_seconds))
