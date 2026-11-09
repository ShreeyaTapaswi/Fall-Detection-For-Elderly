import os
from collections import deque

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from alert_manager import AlertOrchestrator
from motion_analyzer import MotionAnalyzer
from state_machine_standing import StandingFallState, StandingTemporalStateTracker
from transformer_model_standing import StandingFallTransformer


class StandingFallDetector:
    def __init__(self, model_path: str, transformer_path: str | None = None):
        self.model = YOLO(model_path)
        self.transformer = None
        self.class_names = ["Standing", "Falling", "Fallen"]
        if transformer_path and os.path.exists(transformer_path):
            self.transformer = StandingFallTransformer(num_classes=len(self.class_names))
            self.transformer.load_state_dict(torch.load(transformer_path, map_location="cpu"))
            self.transformer.eval()

        self.kpts_history = deque(maxlen=30)
        self.prev_keypoints = None
        self.motion_analyzer = MotionAnalyzer()
        self.state_tracker = StandingTemporalStateTracker()
        self.alert_orchestrator = AlertOrchestrator()

        self.current_state = "Standing"
        self.alert_level = "NONE"
        self.last_velocity = 0.0
        self.rest_counter = 0
        self.velocity_spike_detected = False
        self._smoothed_velocity = 0.0

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        pred_idx = 0
        confidence = 0.5
        flow_velocity = 0.0

        if len(results.keypoints) > 0:
            keypoints = results.keypoints[0].data.cpu().numpy()[0]
            self.kpts_history.append(keypoints.flatten())

            previous_flow = self.last_velocity
            if len(results.boxes) > 0:
                box = results.boxes[0].xyxy[0].cpu().numpy()
                flow_velocity = self.motion_analyzer.compute_optical_flow(frame, person_box=box)
            decel = max(0.0, (previous_flow - flow_velocity) / (1 / 30.0))
            self.last_velocity = flow_velocity

            kpt_velocity = abs(self.calculate_velocity(keypoints, self.prev_keypoints))
            self.prev_keypoints = keypoints.copy()

            if self.transformer and len(self.kpts_history) == self.kpts_history.maxlen:
                seq_input = torch.tensor(np.array(self.kpts_history), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.transformer(seq_input)
                    probs = torch.softmax(logits, dim=1)
                    confidence_tensor, idx_tensor = torch.max(probs, dim=1)
                    confidence = confidence_tensor.item()
                    pred_idx = idx_tensor.item()
            else:
                pred_idx = self._estimate_posture(keypoints)

            proposed_state = self._refine_state_with_physics(pred_idx, kpt_velocity, decel)
            _, state_name = self.state_tracker.update(proposed_state, confidence, velocity=kpt_velocity, decel=decel)
            self.current_state = state_name
            self.alert_level = self.alert_orchestrator.process_alert(self.state_tracker.get_alert_level(), self.current_state)

        self._visualize(frame, results, flow_velocity)
        return frame, self.current_state

    def _estimate_posture(self, keypoints):
        if self._is_horizontal(keypoints):
            return 2
        return 0

    def _refine_state_with_physics(self, prediction_idx, velocity, deceleration):
        predicted = self.class_names[prediction_idx]

        if velocity > 0.8:
            self.velocity_spike_detected = True

        if predicted == "Standing":
            if 0.25 < velocity < 0.55:
                return StandingFallState.PRE_FALL.value
            self.rest_counter = 0
            if velocity < 0.15:
                self.velocity_spike_detected = False
            return StandingFallState.STANDING.value

        if predicted == "Falling":
            if velocity < 0.2:
                return StandingFallState.STANDING.value
            if velocity < 0.5:
                return StandingFallState.PRE_FALL.value
            return StandingFallState.FALLING.value

        is_horizontal = self._is_horizontal(self.prev_keypoints)
        is_at_rest = velocity < 0.18
        if is_at_rest:
            self.rest_counter += 1
            if self.velocity_spike_detected and self.rest_counter > 10 and is_horizontal:
                return StandingFallState.FALLEN.value
            if self.rest_counter > 10 and not is_horizontal:
                self.velocity_spike_detected = False
                return StandingFallState.STANDING.value
            return self.state_tracker.current_state.value

        self.rest_counter = 0
        if deceleration > 10.0 and velocity < 0.4:
            return StandingFallState.IMPACT.value
        if velocity > 0.45:
            return StandingFallState.FALLING.value
        return StandingFallState.PRE_FALL.value

    def _is_horizontal(self, keypoints):
        if keypoints is None:
            return False
        try:
            y_coords = keypoints[[5, 6, 11, 12, 15, 16], 1]
            x_coords = keypoints[[5, 6, 11, 12, 15, 16], 0]
            height = self._estimate_height(keypoints)
            if height < 10:
                return False
            plane_std = np.std(y_coords)
            spread = np.max(x_coords) - np.min(x_coords)
            return plane_std < height * 0.15 and spread > height * 0.5
        except Exception:
            return False

    def calculate_velocity(self, current_keypoints, prev_keypoints, fps=30):
        if prev_keypoints is None:
            return 0.0
        current_hip = (current_keypoints[11][:2] + current_keypoints[12][:2]) / 2
        previous_hip = (prev_keypoints[11][:2] + prev_keypoints[12][:2]) / 2
        dy = current_hip[1] - previous_hip[1]
        person_height = self._estimate_height(current_keypoints)
        if person_height < 10:
            return 0.0
        dy_m = dy * (1.7 / max(person_height, 1.0))
        raw_velocity = dy_m * fps
        self._smoothed_velocity = 0.3 * raw_velocity + 0.7 * self._smoothed_velocity
        return float(np.clip(self._smoothed_velocity, -5.0, 5.0))

    def _estimate_height(self, keypoints):
        shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
        ankle_y = max(keypoints[15][1], keypoints[16][1])
        return abs(float(ankle_y - shoulder_y))

    def _visualize(self, frame, results, velocity):
        alert_colors = {
            "NONE": (0, 255, 0),
            "YELLOW": (0, 255, 255),
            "ORANGE": (0, 165, 255),
            "RED": (0, 0, 255),
            "CRITICAL_RED": (50, 50, 255),
        }
        color = alert_colors.get(self.alert_level, (255, 255, 255))
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (340, 200), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, "STANDING FALL DETECTION", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        cv2.putText(frame, f"STATE: {self.current_state.upper()}", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2)
        cv2.putText(frame, f"ALERT: {self.alert_level}", (20, 115), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        cv2.putText(frame, f"VELOCITY: {velocity:.2f} m/s", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy()
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)


def run_detection(video_path, model_path, output_path, transformer_path=None):
    detector = StandingFallDetector(model_path, transformer_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        processed_frame, _ = detector.process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()


def run_detection_with_progress(video_path, model_path, output_path, transformer_path=None, progress_callback=None):
    detector = StandingFallDetector(model_path, transformer_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    current_state = "Standing"
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        processed_frame, current_state = detector.process_frame(frame)
        out.write(processed_frame)
        frame_idx += 1

        if progress_callback is not None and total_frames > 0:
            progress_callback(
                int((frame_idx / total_frames) * 100),
                current_state,
                {
                    "frame_index": frame_idx,
                    "total_frames": total_frames,
                    "fps": fps,
                    "timestamp_sec": frame_idx / max(fps, 1),
                    "alert_level": detector.alert_level,
                },
            )

    cap.release()
    out.release()

    if progress_callback is not None:
        progress_callback(
            100,
            current_state,
            {
                "frame_index": frame_idx,
                "total_frames": total_frames,
                "fps": fps,
                "timestamp_sec": frame_idx / max(fps, 1),
                "alert_level": detector.alert_level,
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Run standing-fall detection on a video.")
    parser.add_argument("video")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--transformer", default="transformer_standing_fall_detection.pth")
    parser.add_argument("--output", default="inference_outputs/standing_fall_output.mp4")
    args = parser.parse_args()

    run_detection(args.video, args.pose_model, args.output, args.transformer)
    print(f"Saved standing-fall output to {args.output}")


if __name__ == "__main__":
    main()
