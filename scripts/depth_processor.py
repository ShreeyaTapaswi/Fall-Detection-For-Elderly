import numpy as np

class DepthFusionPipeline:
    def __init__(self, ground_height_mm=0):
        self.ground_height = ground_height_mm
        self.bed_height_range = (500, 750) # mm
        
    def process_depth_frame(self, depth_map):
        """
        Detect ground plane using depth data.
        In a real scenario, we'd use RANSAC on the point cloud.
        """
        # Placeholder ground detection logic
        ground_avg = np.mean(depth_map[-50:, :]) # Bottom of image
        self.ground_height = ground_avg
        return self.ground_height

    def classify_surface(self, kpts_3d):
        """
        Classify if person is on bed or floor based on 3D height.
        kpts_3d: (17, 3) relative to camera
        """
        # Height from ground = (Camera Height - Kpt Y)
        # Using a fixed camera height for this simulation
        avg_height = np.mean(kpts_3d[:, 1])
        
        if avg_height < 150:
            return "floor"
        elif self.bed_height_range[0] <= avg_height <= self.bed_height_range[1]:
            return "bed"
        else:
            return "unknown"

    def filter_fall(self, current_state, surface):
        """False positive elimination using surface context."""
        if current_state == "Fallen" and surface == "bed":
            return "Sleeping" # Person is on the bed, likely just lying down
        return current_state
