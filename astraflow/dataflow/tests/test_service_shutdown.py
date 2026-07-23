from types import SimpleNamespace
from unittest.mock import patch

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
