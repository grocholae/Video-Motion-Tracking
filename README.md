# Video Motion Tracking with Lucas-Kanade Method

## Project Overview
This project was completed for a university Computer Vision course. It is a tool designed for detecting and monitoring key feature points within video sequences. The project utilizes classical CV algorithms to estimate optical flow and visualize the movement trajectories of objects in real-time.

## Tech Stack
* Language: Python 3.x
* Core Library: OpenCV
* Data Processing: NumPy

## Key Features
* CLI Integration: Supports loading video files directly via command-line arguments for flexible processing.
* Feature Detection: Implementation of the Shi-Tomasi algorithm (goodFeaturesToTrack) to identify the most stable points for tracking.
* Object Tracking: Utilizes the Lucas-Kanade method to calculate sparse optical flow between consecutive frames.
* Trajectory Visualization: Dynamic drawing of motion paths directly onto the video frames, providing an intuitive view of object displacement over time.

## Usage
Run the script from the terminal by providing the path to your video file:
```bash
python motion_tracking.py --video path_to_video.mp4
