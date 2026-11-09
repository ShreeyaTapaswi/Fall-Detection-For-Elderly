import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from collections import deque
import time

# SOTA Modules
from motion_analyzer import MotionAnalyzer
from state_machine import TemporalStateTracker
from depth_processor import DepthFusionPipeline
from alert_manager import AlertOrchestrator

class AdvancedFallDetector:
    def __init__(self, model_path, transformer_path=None):
        self.model = YOLO(model_path)
        self.transformer = None
        if transformer_path and os.path.exists(transformer_path):
            from transformer_model import FallTransformer
            self.transformer = FallTransformer(17, 128, 4, 2, 512, 3)
            try:
                self.transformer.load_state_dict(torch.load(transformer_path, map_location='cpu'))
                self.transformer.eval()
                print(f"✅ Loaded SOTA Transformer from {transformer_path}")
            except Exception as e:
                print(f"⚠️ Transformer load failed: {e}")

        # SOTA Tracking Components
        self.kpts_history = deque(maxlen=30)
        self.velocity_history = deque(maxlen=30)
        self.height_history = deque(maxlen=30)
        self.prev_keypoints = None
        self._smoothed_velocity = 0.0

        self.motion_analyzer = MotionAnalyzer()
        self.state_tracker = TemporalStateTracker()
        self.depth_processor = DepthFusionPipeline()
        self.alert_orchestrator = AlertOrchestrator()
        
        self.current_state = "Sleeping"
        self.alert_level = "NONE"
        self.last_velocity = 0.0
        self.window_size = 30
        
        # Phase 6: Orientation & Displacement Tracking
        self.baseline_hip_y = None
        self.rest_counter = 0
        self.velocity_spike_detected = False

    def process_frame(self, frame):
        # 1. Faster YOLO Unified Inference (verbose=False)
        results = self.model(frame, verbose=False)[0]
        
        # Default predictions
        pred_idx = 0 
        conf = 0.5
        v_mag = 0.0
        
        if len(results.keypoints) > 0:
            kpts_obj = results.keypoints[0]
            kpts_data = kpts_obj.data.cpu().numpy()[0] 
            self.kpts_history.append(kpts_data.flatten())
            
            # 2. Optimized: Compute Flow & Deceleration
            v_prev = self.last_velocity
            if results.boxes:
                box = results.boxes[0].xyxy[0].cpu().numpy()
                v_mag = self.motion_analyzer.compute_optical_flow(frame, person_box=box)
            
            # SOTA Impact Detection: Decel = (v_prev - v_mag) / dt
            dt = 1/30.0 # Assuming 30fps
            decel = max(0, (v_prev - v_mag) / dt)
            self.last_velocity = v_mag
            
            # Physics-based Velocity from Keypoints
            v_physics = self.calculate_velocity(kpts_data, self.prev_keypoints)
            self.velocity_history.append(abs(v_physics))
            self.prev_keypoints = kpts_data.copy()

            # 3. Faster Transformer Inference
            if self.transformer and len(self.kpts_history) == self.window_size:
                seq_input = torch.tensor(np.array(self.kpts_history)).float().unsqueeze(0)
                with torch.no_grad():
                    logits = self.transformer(seq_input)
                    probs = torch.softmax(logits, dim=1)
                    conf_val, idx_val = torch.max(probs, dim=1)
                    conf = conf_val.item()
                    pred_idx = idx_val.item()
            
            # 4. SOTA Label Mapping & Physics Refinement (Fix 1 & 2)
            fsm_state_idx = self._refine_state_with_physics(pred_idx, abs(v_physics), decel)
            
            from state_machine import FallState
            state_idx, state_name = self.state_tracker.update(
                fsm_state_idx, conf, velocity=abs(v_physics), decel=decel
            )
            
            # 5. Depth & Surface Filter
            surface = self.depth_processor.classify_surface(kpts_data[:, :2])
            self.current_state = self.depth_processor.filter_fall(state_name, surface)
            
            # 6. Alert Orchestration
            self.alert_level = self.alert_orchestrator.process_alert(
                self.state_tracker.get_alert_level(), self.current_state
            )

        # 6. Optimized SOTA Visualization
        self._visualize(frame, results, v_mag)
        
        return frame, self.current_state

    def _visualize(self, frame, results, v_mag):
        # UI Styling based on Alert Severity
        alert_colors = {
            "NONE": (0, 255, 0),         # Green
            "YELLOW": (0, 255, 255),     # Yellow
            "ORANGE": (0, 165, 255),      # Orange
            "RED": (0, 0, 255),          # Red (Impact)
            "CRITICAL_RED": (50, 50, 255) # Deep Red (Fallen)
        }
        ui_color = alert_colors.get(self.alert_level, (255, 255, 255))
        
        # SOTA HUD (Heads-Up Display) - Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 210), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(frame, "SOTA 2025 FALL ARREST", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.putText(frame, f"STATE: {self.current_state.upper()}", (20, 75), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, ui_color, 2)
        
        cv2.putText(frame, f"ALERT: {self.alert_level}", (20, 115), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, ui_color, 2)
        
        cv2.putText(frame, f"VELOCITY: {v_mag:.2f} m/s", (20, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw box and skeleton
        if len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), ui_color, 2)
            
            # Draw impact indicator if in IMPACT state
            if self.current_state == "IMPACT":
                cv2.putText(frame, "!!! IMPACT DETECTED !!!", (int(box[0]), int(box[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    def _refine_state_with_physics(self, model_prediction, velocity, deceleration):
        """Refine model prediction using physics and orientation signatures (SOTA Phase 6)."""
        from state_machine import FallState
        
        # 1. Update Baseline (Sleeping in Bed)
        if self.current_state == "Sleeping" and velocity < 0.1:
            curr_hip = self._get_hip_y(self.prev_keypoints) if self.prev_keypoints is not None else None
            if curr_hip:
                if self.baseline_hip_y is None: self.baseline_hip_y = curr_hip
                else: self.baseline_hip_y = 0.9 * self.baseline_hip_y + 0.1 * curr_hip # EMA

        # 2. Track Velocity Spike (Acceleration phase)
        if velocity > 0.8:
            self.velocity_spike_detected = True
        
        # Case scenarios
        if model_prediction == 0: # SLEEPING
            if velocity > 0.3 and velocity < 0.6: return FallState.PRE_FALL.value
            return FallState.SLEEPING.value
        
        elif model_prediction == 1: # FALLING
            if velocity < 0.15: return FallState.SLEEPING.value
            elif velocity < 0.5: return FallState.PRE_FALL.value
            return FallState.FALLING.value
        
        elif model_prediction == 2: # FALLEN
            # SOTA Phase 6: Multi-metric Signature for FALLEN
            is_at_rest = velocity < 0.15
            is_horizontal = self._is_horizontal(self.prev_keypoints)
            is_displaced = self._is_displaced_downward(self.prev_keypoints)
            
            if is_at_rest:
                self.rest_counter += 1
                # Signature: Spike Detected -> At Rest -> Horizontal -> Displaced Downward
                if self.velocity_spike_detected and self.rest_counter > 15:
                    if is_horizontal and is_displaced:
                        return FallState.FALLEN.value
                    else:
                        # Rest but not horizontal/displaced = likely sitting or slow movement
                        self.rest_counter = 0 
                        return FallState.SLEEPING.value
                return self.state_tracker.current_state.value # Stay in current while confirming
            else:
                self.rest_counter = 0 # Reset if moving
                if velocity > 0.5: return FallState.FALLING.value
            
            return FallState.FALLEN.value if is_horizontal else FallState.FALLING.value
            
        return FallState.SLEEPING.value

    def _is_horizontal(self, kpts):
        """Check if skeleton is horizontal using height-normalized plane threshold."""
        if kpts is None: return False
        try:
            # Pivot points: Shoulders (5,6), Hips (11,12), Ankles (15,16)
            y_coords = kpts[[5, 6, 11, 12, 15, 16], 1]
            x_coords = kpts[[5, 6, 11, 12, 15, 16], 0]
            
            height = self._estimate_height(kpts)
            if height < 10: return False
            
            # Plane Threshold: 15% of body height
            plane_std = np.std(y_coords)
            is_planar = plane_std < (height * 0.15)
            
            # Spread check: X-axis spread should be at least 50% of height
            is_spread = (np.max(x_coords) - np.min(x_coords)) > (height * 0.5)
            
            return is_planar and is_spread
        except: return False

    def _is_displaced_downward(self, kpts):
        """Check if hip is significantly lower than bed baseline."""
        if kpts is None or self.baseline_hip_y is None: return True
        curr_hip = self._get_hip_y(kpts)
        height = self._estimate_height(kpts)
        # Displacement Threshold: 20% of body height lower than bed
        return curr_hip > (self.baseline_hip_y + height * 0.2)

    def _get_hip_y(self, kpts):
        return (kpts[11][1] + kpts[12][1]) / 2

    def calculate_velocity(self, current_keypoints, prev_keypoints, fps=30):
        if prev_keypoints is None: return 0.0
        c_hip = (current_keypoints[11][:2] + current_keypoints[12][:2]) / 2
        p_hip = (prev_keypoints[11][:2] + prev_keypoints[12][:2]) / 2
        dy = c_hip[1] - p_hip[1]
        p_height = self._estimate_height(current_keypoints)
        if p_height < 10: return 0.0
        dy_m = dy * (1.7 / max(1, p_height))
        raw_v = dy_m * fps
        
        # EMA Smoothing
        alpha = 0.3
        self._smoothed_velocity = alpha * raw_v + (1 - alpha) * self._smoothed_velocity
        return np.clip(self._smoothed_velocity, -5.0, 5.0)
    
    def _estimate_height(self, keypoints):
        shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
        ankle_y = max(keypoints[15][1], keypoints[16][1])
        return abs(ankle_y - shoulder_y)

def run_detection(video_path, model_path, output_path, transformer_path=None):
    detector = AdvancedFallDetector(model_path, transformer_path)
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Running inference on {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, state = detector.process_frame(frame)
        out.write(processed_frame)
        
    cap.release()
    out.release()
    print(f"✅ Saved results to {output_path}")


def run_detection_with_progress(video_path, model_path, output_path,
                                transformer_path=None, progress_callback=None):
    """
    Same as run_detection but calls progress_callback(pct: int, state: str)
    on every processed frame so callers (e.g. the FastAPI server) can stream
    live progress updates.
    """
    detector = AdvancedFallDetector(model_path, transformer_path)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    current_state = "Sleeping"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, current_state = detector.process_frame(frame)
        out.write(processed_frame)
        frame_idx += 1

        if progress_callback is not None and total_frames > 0:
            pct = int((frame_idx / total_frames) * 100)
            progress_callback(pct, current_state)

    cap.release()
    out.release()

    # Final 100% callback
    if progress_callback is not None:
        progress_callback(100, current_state)

    print(f"✅ Saved results to {output_path}")

if __name__ == "__main__":
    # Point to the SOTA 2025 trained models
    best_yolo = 'yolo11n-pose.pt' 
    best_transformer = 'transformer_fall_detection.pth'
    
    # List of videos to test
    videos = ['test_video_1.mp4', 'test_video_2.mp4', 'test_video_3.mp4', 'test_video_4.mp4']
    
    output_dir = "inference_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")

    for video in videos:
        if not os.path.exists(video):
            print(f"⚠️ Video not found: {video}")
            continue
            
        output_path = os.path.join(output_dir, f"output_SOTA_{video}")
        run_detection(video, best_yolo, output_path, best_transformer)
