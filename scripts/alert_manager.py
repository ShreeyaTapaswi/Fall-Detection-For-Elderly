import time
import json

class AlertOrchestrator:
    def __init__(self):
        # SOTA Alert Severity Levels
        self.severity_levels = {
            "NONE": 0,          # Sleeping
            "YELLOW": 1,        # Pre-Fall (Warning)
            "ORANGE": 2,        # Falling (Immediate)
            "RED": 3,           # Impact (Critical)
            "CRITICAL_RED": 4   # Fallen (Persistent)
        }
        self.current_level = "NONE"
        self.last_alert_time = 0
        self.cooldown = 30 # seconds
        
    def process_alert(self, level, state_name):
        """
        Processes an alert and triggers necessary actions.
        """
        if self.severity_levels[level] <= self.severity_levels[self.current_level]:
            # Downgrade or same level (unless significant time passed)
            if time.time() - self.last_alert_time > self.cooldown:
                self.current_level = level
        else:
            # Upgrade severity
            self.current_level = level
            self._trigger_action(level, state_name)
            
        return self.current_level

    def _trigger_action(self, level, state):
        """Internal triggers for external side effects."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if level == "YELLOW":
            print(f"[{timestamp}] 🟡 PRE-FALL WARNING: {state}")
        elif level == "ORANGE":
            print(f"[{timestamp}] 🟠 FALL DETECTED (VOICE CHECK): {state}")
            # In a real app, play: "Are you okay?"
        elif level == "RED":
            print(f"[{timestamp}] 🔴 IMPACT DETECTED (CONFIRM FALL): {state}")
            self._send_notification(state, "SMS/Mobile App")
        elif level == "CRITICAL_RED":
            print(f"[{timestamp}] 🔥 FALLEN: NO RECOVERY FOR 10s. EMERGENCY DISPATCH.")
            self._send_notification(state, "Emergency Services")

    def _send_notification(self, state, channel):
        # Mock notification logic
        alert_payload = {
            "event": "Fall Detected",
            "state": state,
            "channel": channel,
            "timestamp": time.time()
        }
        with open("alerts_log.json", "a") as f:
            f.write(json.dumps(alert_payload) + "\n")

if __name__ == "__main__":
    orchestrator = AlertOrchestrator()
    orchestrator.process_alert("RED", "Fallen")
