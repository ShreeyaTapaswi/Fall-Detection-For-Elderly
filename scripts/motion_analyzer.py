import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        self.prev_gray = None
        self.prev_y_avg = None
        self.prev_head_pos = None

    def compute_optical_flow(self, frame, person_box=None):
        """Compute vertical velocity magnitude using downsampled Farneback flow."""
        # Optimization: Downsample the frame specifically for flow calc
        scale = 0.25
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
            
        # Standard parameters, but on a smaller image
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 10, 3, 5, 1.1, 0)
        v_flow = flow[..., 1]
        self.prev_gray = gray
        
        # If we have a person box, only look at motion within that area
        # (scaled accordingly)
        if person_box is not None:
             x1, y1, x2, y2 = (person_box * scale).astype(int)
             v_flow = v_flow[max(0, y1):min(v_flow.shape[0], y2), 
                             max(0, x1):min(v_flow.shape[1], x2)]
        
        if np.any(v_flow > 0):
            return np.mean(v_flow[v_flow > 0]) * (1/scale) # Scale velocity back up
        return 0.0

    def compute_skeletal_features(self, keypoints, confs):
        """
        Compute torso angle, aspect ratio, limb spread, etc.
        keypoints: (17, 2), confs: (17,)
        """
        valid_points = keypoints[confs > 0.5]
        if len(valid_points) < 2:
            return np.zeros(10) # Fallback

        # 1. Bounding Box Aspect Ratio
        x_min, y_min = np.min(valid_points, axis=0)
        x_max, y_max = np.max(valid_points, axis=0)
        width = x_max - x_min
        height = max(1, y_max - y_min)
        aspect_ratio = width / height

        # 2. Torso Angle (Shoulders to Hips)
        angle = 0.0
        if confs[5] > 0.5 and confs[6] > 0.5 and confs[11] > 0.5 and confs[12] > 0.5:
            mid_shoulder = (keypoints[5] + keypoints[6]) / 2
            mid_hip = (keypoints[11] + keypoints[12]) / 2
            dx, dy = mid_hip[0] - mid_shoulder[0], mid_hip[1] - mid_shoulder[1]
            angle = np.degrees(np.arctan2(dy, dx))

        # 3. Head Velocity
        head_vel = 0.0
        if confs[0] > 0.5:
            current_head = keypoints[0]
            if self.prev_head_pos is not None:
                head_vel = np.linalg.norm(current_head - self.prev_head_pos) * self.fps
            self.prev_head_pos = current_head

        # 4. Center of Mass (Mean of keypoints)
        com = np.mean(valid_points, axis=0)
        
        # Feature vector
        return np.array([
            aspect_ratio, angle, head_vel, 
            com[0], com[1], width, height,
            np.std(valid_points[:, 0]), np.std(valid_points[:, 1]),
            len(valid_points)
        ])

if __name__ == "__main__":
    # Test stub
    analyzer = MotionAnalyzer()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    v_mag = analyzer.compute_optical_flow(dummy_frame)
    print(f"Optical Flow Magnitude: {v_mag}")
