from datetime import datetime, timezone

from GridPythia.config import AppConfig
from GridPythia.server.scheduler import (
    _adaptive_dispatch_buffer_seconds,
    _next_scheduler_trigger,
    _scheduler_lead_seconds,
)


class TestSchedulerTiming:
    def test_lead_uses_solver_time_limit_plus_buffer(self):
        cfg = AppConfig.from_dict({
            "prediction": {"horizon": 24},
            "optimization": {
                "solver": {"solver_opts": {"time_limit": 18}},
                "batteries": [],
                "inverters": [],
            },
            "server": {"scheduler": {"dispatch_buffer_seconds": 5}},
        })

        assert _scheduler_lead_seconds(cfg) == 23.0

    def test_late_publish_increases_buffer_up_to_max(self):
        cfg = AppConfig.from_dict({
            "prediction": {"horizon": 24},
            "optimization": {"batteries": [], "inverters": []},
            "server": {
                "scheduler": {
                    "dispatch_buffer_seconds": 5,
                    "dispatch_buffer_max_seconds": 30,
                }
            },
        })

        assert _adaptive_dispatch_buffer_seconds(cfg, publish_lateness_s=9) == 14.0
        assert _adaptive_dispatch_buffer_seconds(cfg, publish_lateness_s=50) == 30.0

    def test_next_trigger_runs_before_upcoming_slot(self):
        cfg = AppConfig.from_dict({
            "prediction": {"horizon": 24},
            "optimization": {
                "solver": {"solver_opts": {"time_limit": 30}},
                "batteries": [],
                "inverters": [],
            },
            "server": {
                "scheduler": {
                    "optimization_interval_minutes": 15,
                    "dispatch_buffer_seconds": 5,
                }
            },
        })
        now = datetime(2026, 4, 29, 16, 0, 0, tzinfo=timezone.utc)

        slot, run_at, lead_s = _next_scheduler_trigger(now, cfg)

        assert slot == datetime(2026, 4, 29, 16, 15, 0, tzinfo=timezone.utc)
        assert run_at == datetime(2026, 4, 29, 16, 14, 25, tzinfo=timezone.utc)
        assert lead_s == 35.0

    def test_next_trigger_uses_next_slot_after_completed_cycle(self):
        cfg = AppConfig.from_dict({
            "prediction": {"horizon": 24},
            "optimization": {"batteries": [], "inverters": []},
            "server": {"scheduler": {"optimization_interval_minutes": 15}},
        })
        now = datetime(2026, 4, 29, 16, 14, 50, tzinfo=timezone.utc)
        last_slot = datetime(2026, 4, 29, 16, 15, 0, tzinfo=timezone.utc)

        slot, _, _ = _next_scheduler_trigger(now, cfg, last_dispatch_slot=last_slot)

        assert slot == datetime(2026, 4, 29, 16, 30, 0, tzinfo=timezone.utc)