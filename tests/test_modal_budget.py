import pytest

from qchunk.modal_budget import ModalResources, budget_timeout_seconds, estimate_cost


def test_a10_resource_rate_includes_gpu_cpu_and_memory():
    resources = ModalResources(gpu="A10", cpu_cores=4, memory_gib=32)

    assert resources.usd_per_hour == pytest.approx(1.545984)
    assert estimate_cost(resources, 3600) == pytest.approx(resources.usd_per_hour)


def test_budget_timeout_caps_spend_and_platform_duration():
    resources = ModalResources(gpu="A10", cpu_cores=4, memory_gib=32)
    timeout = budget_timeout_seconds(resources, 27.0)

    assert timeout < 24 * 60 * 60
    assert estimate_cost(resources, timeout) <= 27.0
    assert estimate_cost(resources, timeout + 1) > 27.0


def test_budget_rejects_unknown_gpu_and_invalid_values():
    with pytest.raises(ValueError, match="unsupported GPU"):
        ModalResources(gpu="RTX-UNKNOWN").validate()
    with pytest.raises(ValueError, match="cannot be negative"):
        estimate_cost(ModalResources(), -1)
    with pytest.raises(ValueError, match="must be positive"):
        budget_timeout_seconds(ModalResources(), 0)
