import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from astraflow.dataflow import service as service_module


def test_shutdown_stops_dataflow_before_raas() -> None:
    events: list[str] = []

    fake_service = SimpleNamespace(
        stop_all=lambda: events.append("stop_dataflow"),
        raas_pool=SimpleNamespace(
            shutdown_all=lambda **_kwargs: events.append("shutdown_raas") or {}
        ),
    )

    class DeferredExitThread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self) -> None:
            events.append("schedule_exit")

    with (
        service_module.app.app_context(),
        patch.object(service_module, "_get_service", return_value=fake_service),
        patch.object(service_module.threading, "Thread", DeferredExitThread),
    ):
        response = service_module.shutdown()

    assert response.status_code == 200
    assert events == ["stop_dataflow", "shutdown_raas", "schedule_exit"]


def test_sync_notify_loads_weights_before_advancing_version() -> None:
    events: list[str] = []
    service = service_module.AstraFlowService.__new__(
        service_module.AstraFlowService
    )
    service._model_versions = {("default", "model0"): 0}
    service._registered_model_ids = {"default": {"model0"}}
    service._pending_versions = {}
    service._pending_eval = {}
    service._version_barrier_cond = threading.Condition()
    service._version_barrier_generation = 0
    service._barrier_eval_results = None
    service._balance_report_dir = None
    service._balance_report_freq = 0
    service._balance_last_saved_version = 0
    service.versions = {"default": 0}
    service.flows = {
        "default": SimpleNamespace(
            data_acquisition=SimpleNamespace(
                notify_version_changed=lambda version: events.append(
                    f"curator:{version}"
                )
            )
        )
    }
    service.raas_pool = SimpleNamespace(
        set_version_local=lambda version: events.append(f"pool:{version}")
    )

    def load_weights(agent_name, model_id, version):
        assert service._model_versions[(agent_name, model_id)] == 0
        events.append(f"load:{version}")
        return {"worker": {"ok": True, "version": version}}

    service._trigger_raas_weight_load_single = load_weights

    service.notify_version(
        "default",
        version=1,
        model_id="model0",
        sync_weight_load=True,
    )

    assert service._model_versions[("default", "model0")] == 1
    assert service.versions["default"] == 1
    assert events == ["load:1", "pool:1", "curator:1"]


def test_sync_notify_does_not_advance_version_when_load_fails() -> None:
    service = service_module.AstraFlowService.__new__(
        service_module.AstraFlowService
    )
    service._model_versions = {("default", "model0"): 0}
    service._trigger_raas_weight_load_single = lambda *_args: {
        "worker": {"ok": False, "error": "load failed"}
    }

    with pytest.raises(RuntimeError, match="Synchronous weight load failed"):
        service.notify_version(
            "default",
            version=1,
            model_id="model0",
            sync_weight_load=True,
        )

    assert service._model_versions[("default", "model0")] == 0
