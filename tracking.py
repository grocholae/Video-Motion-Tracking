#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np

def load_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}", file=sys.stderr)
        return None

    return cap

def detect_points(gray_image: np.ndarray):
    # Using Shi-Tomasi corner detection
    points = cv2.goodFeaturesToTrack(gray_image, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=5, blockSize=5)
    return points

def track_points(prev_gray, curr_gray, prev_points):
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    new_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

    return new_points, status, err

def draw_points(image, points):
    for point in points:
        x, y = point.ravel()
        # Casting to int and saving as a tuple (x, y)
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

    return image

def draw_trajectories(mask, old_points, new_points):
    for old, new in zip(old_points, new_points):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 1)

    return mask

# ============================================================
# Main video processing loop
# ============================================================

def process_video(video_path: str):

    cap = load_video(video_path)
    if cap is None:
        return

    # Read the first frame
    success, first_frame = cap.read()
    if not success:
        print("ERROR: Could not read the first frame of the video", file=sys.stderr)
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial points
    prev_points = detect_points(prev_gray)

    # Mask for drawing trajectories
    trajectory_mask = np.zeros_like(first_frame)

    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track points
        new_points, status, error = track_points(
            prev_gray,
            curr_gray,
            prev_points
        )

        # Filter points based on status (1 = found, 0 = lost)
        good_old_points = prev_points[status == 1]
        good_new_points = new_points[status == 1]

        if len(good_old_points) == 0:
            print("No good points left to track", file=sys.stderr)
            break

        # Draw trajectories
        trajectory_mask = draw_trajectories(
            trajectory_mask,
            good_old_points,
            good_new_points
        )

        # Draw current points
        result = draw_points(frame.copy(), good_new_points)

        # Combine image with trajectory mask
        result = cv2.add(result, trajectory_mask)

        count = len(good_new_points)
        status_text = f"Number of points tracked: {count}"
        cv2.putText(result, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Optical Flow - Point Tracking", result)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27): # 'q' or ESC to exit
            break

        # Update previous frame and points
        prev_gray = curr_gray.copy()
        prev_points = good_new_points.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB4 - Optical Flow, starter template")
    parser.add_argument("--video", required=True, help="Path to the video file")

    args = parser.parse_args()

    process_video(args.video)


if __name__ == "__main__":
    main()