from enum import Enum
from collections import deque
import numpy as np

class FallState(Enum):
    SLEEPING = 0
    PRE_FALL = 1
    FALLING = 2
    IMPACT = 3
    FALLEN = 4

class TemporalStateTracker:
    """SOTA 5-state FSM with physical validation gate (Clean Stable Version)."""
    
    VALID_TRANSITIONS = {
        FallState.SLEEPING: [FallState.PRE_FALL, FallState.FALLING],
        FallState.PRE_FALL: [FallState.FALLING, FallState.SLEEPING],
        FallState.FALLING: [FallState.IMPACT, FallState.SLEEPING, FallState.FALLEN], # Phase 6: Direct Fallen
        FallState.IMPACT: [FallState.FALLEN],
        FallState.FALLEN: [FallState.SLEEPING] # Recovery
    }

    def __init__(self, window_size=10):
        self.current_state = FallState.SLEEPING
        self.history = deque(maxlen=window_size)
        self.impact_detected = False
        
    def update(self, proposed_idx, confidence, velocity=0, decel=0):
        """Update state using physical validation logic."""
        try:
            proposed_state = FallState(proposed_idx)
        except (ValueError, TypeError):
            return self.current_state.value, self.current_state.name

        # 1. Physical Validation Gate
        is_valid, reason = self.validate_transition(self.current_state, proposed_state, velocity, decel)
        
        if is_valid:
            if proposed_state != self.current_state:
                print(f"🔄 State Transition: {self.current_state.name} -> {proposed_state.name} ({reason})")
            self.current_state = proposed_state
            if proposed_state == FallState.IMPACT:
                self.impact_detected = True
            if proposed_state == FallState.SLEEPING:
                self.impact_detected = False # Reset on recovery
            if proposed_state == FallState.FALLEN and not self.impact_detected:
                # Emergency catch-up if impact missed
                self.impact_detected = True
        
        self.history.append(self.current_state)
        return self.current_state.value, self.current_state.name

    def validate_transition(self, current, proposed, velocity, decel):
        """
        SOTA Validation Logic with stricter physics-based rules.
        """
        if current == proposed:
            return True, "Stable"
            
        # Check FSM allowed transitions
        if proposed not in self.VALID_TRANSITIONS.get(current, []):
            # Hard override for fast falls (skip Pre-Fall)
            if current == FallState.SLEEPING and proposed == FallState.FALLING:
                if velocity > 0.7:  # Very high velocity
                    return True, "Fast Fall (skipped Pre-Fall)"
                else:
                    return False, f"Invalid: SLEEPING → FALLING requires velocity >0.7 (got {velocity:.2f})"
            
            return False, f"Invalid State Jump: {current.name} → {proposed.name}"
        
        # Physics-based validation for each transition
        
        # SLEEPING → PRE_FALL
        if current == FallState.SLEEPING and proposed == FallState.PRE_FALL:
            if velocity < 0.2:
                return False, f"PRE_FALL requires velocity >0.2 (got {velocity:.2f})"
            return True, "Destabilization detected"
        
        # PRE_FALL → FALLING
        if current == FallState.PRE_FALL and proposed == FallState.FALLING:
            if velocity < 0.5:
                return False, f"FALLING requires velocity >0.5 (got {velocity:.2f})"
            return True, "Fall initiated"
        
        # FALLING → IMPACT
        if current == FallState.FALLING and proposed == FallState.IMPACT:
            if decel < 12.0:
                return False, f"IMPACT requires decel >12.0 (got {decel:.1f})"
            if velocity > 0.4:  # Slightly safer than 0.3 for noisy data
                return False, f"IMPACT requires velocity <0.4 (got {velocity:.2f})"
            return True, "Impact detected"
        
        # IMPACT / FALLING → FALLEN
        if (current in [FallState.IMPACT, FallState.FALLING]) and proposed == FallState.FALLEN:
            # FALLING -> FALLEN is a "Fast Impact" jump
            return True, "Impact Confirmed"
        
        # FALLEN → SLEEPING (recovery)
        if current == FallState.FALLEN and proposed == FallState.SLEEPING:
            # Allow recovery if person is moving (standing up)
            return True, "Recovery detected"
        
        # PRE_FALL → SLEEPING (abort)
        if current == FallState.PRE_FALL and proposed == FallState.SLEEPING:
            return True, "Pre-fall aborted"
        
        # FALLING → SLEEPING (abort - rare but possible)
        if current == FallState.FALLING and proposed == FallState.SLEEPING:
            if velocity < 0.2:
                return True, "Fall aborted (caught self)"
            return False, "Cannot abort fall at high velocity"
        
        # Default allow if in valid transitions list
        return True, "Valid"

    def get_alert_level(self):
        levels = {
            FallState.SLEEPING: "NONE",
            FallState.PRE_FALL: "YELLOW",
            FallState.FALLING: "ORANGE",
            FallState.IMPACT: "RED",
            FallState.FALLEN: "CRITICAL_RED"
        }
        return levels.get(self.current_state, "NONE")
