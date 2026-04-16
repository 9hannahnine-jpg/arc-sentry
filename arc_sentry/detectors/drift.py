"""
Drift detector — long-horizon session monitoring.
Same D(t) math as SessionMonitor, longer time constants.
Catches context poisoning, slow social engineering, agent trajectory drift.
"""
from arc_sentry.representations.trajectory import SessionMonitor


class DriftMonitor(SessionMonitor):
    """
    Long-horizon session monitor for drift detection.
    Defaults: short=20, long=100, min_history=50.
    """

    def __init__(self, short_horizon=20, long_horizon=100, min_history=50):
        super().__init__(
            short_horizon=short_horizon,
            long_horizon=long_horizon,
            min_history=min_history,
        )

    def push(self, fr_distance):
        result = super().push(fr_distance)
        return {
            "drift_D": result.get("D"),
            "drift_tau": result.get("tau"),
            "drift_lambda": result.get("lambda"),
            "drift_blocked": result.get("session_blocked"),
            "drift_threshold": result.get("threshold"),
            "drift_history_len": result.get("history_len"),
        }
