from datetime import datetime, timezone

from GridPythia.server.plan_utils import stitch_current_slot_from_previous_plan as _stitch_current_slot_from_previous_plan


class TestMqttPlanPublish:
    def test_prepends_current_slot_from_previous_plan_when_new_plan_starts_next_slot(self):
        previous_steps = [
            {"timestamp": "2026-04-29T16:00:00+00:00", "mode": 0, "mode_name": "IDLE"},
            {"timestamp": "2026-04-29T16:15:00+00:00", "mode": 1, "mode_name": "DISCHARGE"},
        ]
        new_steps = [
            {"timestamp": "2026-04-29T16:15:00+00:00", "mode": 0, "mode_name": "IDLE"},
            {"timestamp": "2026-04-29T16:30:00+00:00", "mode": 0, "mode_name": "IDLE"},
        ]

        stitched = _stitch_current_slot_from_previous_plan(
            new_steps,
            previous_steps,
            published_at=datetime(2026, 4, 29, 16, 14, 45, tzinfo=timezone.utc),
            dt_hours=0.25,
        )

        assert [step["timestamp"] for step in stitched] == [
            "2026-04-29T16:00:00+00:00",
            "2026-04-29T16:15:00+00:00",
            "2026-04-29T16:30:00+00:00",
        ]
        assert stitched[0]["mode_name"] == "IDLE"

    def test_does_not_prepend_when_new_plan_already_covers_current_slot(self):
        previous_steps = [{"timestamp": "2026-04-29T16:00:00+00:00", "mode": 0}]
        new_steps = [{"timestamp": "2026-04-29T16:00:00+00:00", "mode": 0}]

        stitched = _stitch_current_slot_from_previous_plan(
            new_steps,
            previous_steps,
            published_at=datetime(2026, 4, 29, 16, 0, 5, tzinfo=timezone.utc),
            dt_hours=0.25,
        )

        assert stitched == new_steps

    def test_leaves_new_plan_unchanged_when_previous_slot_missing(self):
        previous_steps = [{"timestamp": "2026-04-29T15:45:00+00:00", "mode": 0}]
        new_steps = [{"timestamp": "2026-04-29T16:15:00+00:00", "mode": 0}]

        stitched = _stitch_current_slot_from_previous_plan(
            new_steps,
            previous_steps,
            published_at=datetime(2026, 4, 29, 16, 14, 45, tzinfo=timezone.utc),
            dt_hours=0.25,
        )

        assert stitched == new_steps