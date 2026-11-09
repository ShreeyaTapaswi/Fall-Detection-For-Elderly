import numpy as np
import os

def generate_synthetic_sequences(num_sequences=5000, seq_len=30):
    """
    Generate synthetic keypoint sequences simulating Sleeping, Falling, and Fallen states.
    Keypoints: 17 points * (x, y, conf) = 51 features.
    """
    X = []
    y = []

    print(f"Generating {num_sequences} synthetic sequences...")

    for _ in range(num_sequences):
        state = np.random.randint(0, 5) # 0:Sleeping, 1:Pre-Fall, 2:Falling, 3:Impact, 4:Fallen
        seq = []
        
        # Base keypoints (normalized 0-1)
        base_kpts = np.random.rand(17, 3)
        base_kpts[:, 2] = np.random.uniform(0.7, 1.0, 17) # High confidence
        
        for t in range(seq_len):
            current_kpts = base_kpts.copy()
            progress = t / seq_len
            
            if state == 0: # Sleeping (Static or minor jitter)
                current_kpts[:, :2] += np.random.normal(0, 0.005, (17, 2))
                current_kpts[:, 1] += 0.1 # On bed
            
            elif state == 1: # Pre-Fall (Rolling, destabilization)
                # Roll towards edge (x shift)
                current_kpts[:, 0] += progress * 0.15 
                current_kpts[:, 1] += np.random.normal(0, 0.01, 17) # Small y oscillation
            
            elif state == 2: # Falling (Rapid downward movement)
                # Rapid y increase
                current_kpts[:, 1] += progress * 0.5 
                # Lateral spread
                center_x = np.mean(current_kpts[:, 0])
                current_kpts[:, 0] = center_x + (current_kpts[:, 0] - center_x) * (1 + progress * 0.8)
            
            elif state == 3: # Impact (Sudden stop + recoil)
                # Reach ground level (high y)
                current_kpts[:, 1] += 0.55
                # Brief recoil at the start, then static
                recoil = max(0, 0.05 * (1 - progress * 5))
                current_kpts[:, 1] -= recoil
                current_kpts[:, :2] += np.random.normal(0, 0.02, (17, 2))
            
            elif state == 4: # Fallen (Static at low height)
                current_kpts[:, 1] += 0.6 # Lying flat on floor
                current_kpts[:, :2] += np.random.normal(0, 0.005, (17, 2))
                
            seq.append(current_kpts.flatten())
        
        X.append(np.array(seq))
        y.append(state)

    X = np.array(X)
    y = np.array(y)
    
    np.save("X_synthetic.npy", X)
    np.save("y_synthetic.npy", y)
    print(f"✅ Generated {len(X)} synthetic samples. Saved to X_synthetic.npy and y_synthetic.npy")

if __name__ == "__main__":
    generate_synthetic_sequences()
