from collections import deque
from enum import Enum


class StandingFallState(Enum):
    STANDING = 0
    PRE_FALL = 1
    FALLING = 2
    IMPACT = 3
    FALLEN = 4


class StandingTemporalStateTracker:
    VALID_TRANSITIONS = {
        StandingFallState.STANDING: [StandingFallState.PRE_FALL, StandingFallState.FALLING],
        StandingFallState.PRE_FALL: [StandingFallState.STANDING, StandingFallState.FALLING],
        StandingFallState.FALLING: [StandingFallState.IMPACT, StandingFallState.FALLEN, StandingFallState.STANDING],
        StandingFallState.IMPACT: [StandingFallState.FALLEN],
        StandingFallState.FALLEN: [StandingFallState.STANDING],
    }

    def __init__(self, window_size: int = 10):
        self.current_state = StandingFallState.STANDING
        self.history = deque(maxlen=window_size)
        self.impact_detected = False

    def update(self, proposed_idx: int, confidence: float, velocity: float = 0.0, decel: float = 0.0):
        try:
            proposed_state = StandingFallState(proposed_idx)
        except ValueError:
            return self.current_state.value, self.current_state.name.title().replace("_", " ")

        is_valid, _ = self.validate_transition(self.current_state, proposed_state, velocity, decel, confidence)
        if is_valid:
            self.current_state = proposed_state
            if proposed_state == StandingFallState.IMPACT:
                self.impact_detected = True
            if proposed_state == StandingFallState.STANDING:
                self.impact_detected = False
            if proposed_state == StandingFallState.FALLEN:
                self.impact_detected = True

        self.history.append(self.current_state)
        return self.current_state.value, self.current_state.name.title().replace("_", " ")

    def validate_transition(self, current, proposed, velocity, decel, confidence):
        if current == proposed:
            return True, "stable"

        if proposed not in self.VALID_TRANSITIONS.get(current, []):
            if current == StandingFallState.STANDING and proposed == StandingFallState.FALLING:
                return velocity > 0.7, "fast fall"
            return False, "invalid jump"

        if proposed == StandingFallState.PRE_FALL:
            return velocity > 0.2, "destabilized"
        if proposed == StandingFallState.FALLING:
            return velocity > 0.45 or confidence > 0.75, "falling"
        if proposed == StandingFallState.IMPACT:
            return decel > 10.0 and velocity < 0.4, "impact"
        if proposed == StandingFallState.FALLEN:
            return self.impact_detected or current in [StandingFallState.FALLING, StandingFallState.IMPACT], "fallen"
        if proposed == StandingFallState.STANDING:
            return True, "recovered"
        return True, "valid"

    def get_alert_level(self) -> str:
        return {
            StandingFallState.STANDING: "NONE",
            StandingFallState.PRE_FALL: "YELLOW",
            StandingFallState.FALLING: "ORANGE",
            StandingFallState.IMPACT: "RED",
            StandingFallState.FALLEN: "CRITICAL_RED",
        }.get(self.current_state, "NONE")
