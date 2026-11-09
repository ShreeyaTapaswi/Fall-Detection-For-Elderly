import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from standing_dataset import CLASS_NAMES, StandingDatasetPair, discover_standing_dataset_pairs, parse_cvat_frame_labels


def extract_sequences(pair: StandingDatasetPair, model: YOLO, seq_len: int = 30) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(pair.video_path))
    frame_labels = parse_cvat_frame_labels(pair.annotation_path)

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    current_sequence: list[np.ndarray] = []
    current_labels: list[int] = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {pair.source_name} from {pair.video_path.name}...")

    for frame_idx in tqdm(range(total_frames)):
        ok, frame = cap.read()
        if not ok:
            break

        label = frame_labels.get(frame_idx)
        if label is None:
            current_sequence.clear()
            current_labels.clear()
            continue

        results = model(frame, verbose=False)[0]
        if len(results.keypoints) == 0:
            current_sequence.clear()
            current_labels.clear()
            continue

        keypoints = results.keypoints[0].data.cpu().numpy()[0].flatten().astype(np.float32)
        current_sequence.append(keypoints)
        current_labels.append(label)

        if len(current_sequence) == seq_len:
            sequences.append(np.stack(current_sequence))
            labels.append(int(np.bincount(np.array(current_labels), minlength=len(CLASS_NAMES)).argmax()))
            current_sequence = current_sequence[1:]
            current_labels = current_labels[1:]

    cap.release()

    if not sequences:
        return np.empty((0, seq_len, 51), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(sequences).astype(np.float32), np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Extract standing-fall transformer sequences.")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--output-dir", default="artifacts/standing_transformer")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    pairs = discover_standing_dataset_pairs(project_root)
    if not pairs:
        print("No matching standing video/XML pairs found.")
        return

    model = YOLO(args.pose_model)
    all_sequences: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for pair in pairs:
        sequences, labels = extract_sequences(pair, model, seq_len=args.seq_len)
        if len(sequences) == 0:
            print(f"Skipped {pair.source_name}: no valid sequences.")
            continue
        all_sequences.append(sequences)
        all_labels.append(labels)

    if not all_sequences:
        print("No standing sequences were extracted.")
        return

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    X = np.concatenate(all_sequences, axis=0)
    y = np.concatenate(all_labels, axis=0)

    np.save(output_dir / "X_sequences.npy", X)
    np.save(output_dir / "y_labels.npy", y)
    (output_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    print(f"Saved {len(X)} standing-fall sequences to {output_dir}")


if __name__ == "__main__":
    main()
