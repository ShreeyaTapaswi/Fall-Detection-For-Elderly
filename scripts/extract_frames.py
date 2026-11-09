import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, video_id):
    """
    Extract frames from a video file and save them as images.
    Naming convention: v{video_id}_f{frame_count:06d}.jpg
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting frames from {video_path}...")
    frame_count = 0
    extracted_count = 0
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_filename = os.path.join(output_dir, f"v{video_id}_f{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            extracted_count += 1
            pbar.update(1)
            
    cap.release()
    print(f"✅ Extracted {extracted_count} frames to {output_dir}")
    return extracted_count

if __name__ == "__main__":
    # Define paths
    base_dir = Path("c:/Users/ADMIN/Documents/Fall_Detection_2")
    output_images_dir = base_dir / "dataset/images/train" # Initial extraction to train, will split later
    
    videos = [
        {"path": base_dir / "train_video_1.mp4", "id": 1},
        {"path": base_dir / "train_video_2.mp4", "id": 2}
    ]
    
    total_extracted = 0
    for video in videos:
        total_extracted += extract_frames(str(video["path"]), str(output_images_dir), video["id"])
    
    print(f"\nTotal frames extracted across all videos: {total_extracted}")
