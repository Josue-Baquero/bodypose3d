import cv2
import numpy as np
import sys
from ultralytics import YOLO
from utils import DLT, get_projection_matrix, write_keypoints_to_disk

frame_shape = [720, 1280]

# Updated YOLO11 keypoints indices based on COCO format
pose_keypoints = {
    'right_wrist': 10,    # Right wrist
    'right_elbow': 8,     # Right elbow
    'right_shoulder': 6,  # Right shoulder
    'left_shoulder': 5,   # Left shoulder
    'left_elbow': 7,      # Left elbow
    'left_wrist': 9,      # Left wrist
    'right_hip': 12,      # Right hip
    'left_hip': 11,       # Left hip
    'right_knee': 14,     # Right knee
    'left_knee': 13,      # Left knee
    'right_ankle': 16,    # Right ankle
    'left_ankle': 15      # Left ankle
}

def run_yolo(input_stream1, input_stream2, P0, P1):
    # Initialize YOLO11 model for pose estimation
    model = YOLO('yolo11n-pose.pt')
    
    # Input video streams
    cap0 = cv2.VideoCapture(input_stream1)
    cap1 = cv2.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # Set camera resolution
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # Containers for detected keypoints
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    
    while True:
        # Read frames from streams
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: 
            break

        # Crop to 720x720 to match calibration parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:, frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            frame1 = frame1[:, frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # Run YOLO pose estimation on both frames
        results0 = model(frame0, verbose=False)[0]
        results1 = model(frame1, verbose=False)[0]

        # Process keypoints from camera 0
        frame0_keypoints = []
        if len(results0.keypoints.data) > 0:
            keypoints = results0.keypoints.data[0]  # Get first person's keypoints
            for kpt_id in pose_keypoints.values():
                if kpt_id < len(keypoints):
                    x, y = int(keypoints[kpt_id][0]), int(keypoints[kpt_id][1])
                    conf = float(keypoints[kpt_id][2])
                    if conf > 0.5:  # Confidence threshold
                        cv2.circle(frame0, (x, y), 3, (0,0,255), -1)
                        cv2.putText(frame0, str(kpt_id), (x+5, y+5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        frame0_keypoints.append([x, y])
                    else:
                        frame0_keypoints.append([-1, -1])
                else:
                    frame0_keypoints.append([-1, -1])
        else:
            frame0_keypoints = [[-1, -1]] * len(pose_keypoints)
            
        kpts_cam0.append(frame0_keypoints)

        # Process keypoints from camera 1
        frame1_keypoints = []
        if len(results1.keypoints.data) > 0:
            keypoints = results1.keypoints.data[0]  # Get first person's keypoints
            for kpt_id in pose_keypoints.values():
                if kpt_id < len(keypoints):
                    x, y = int(keypoints[kpt_id][0]), int(keypoints[kpt_id][1])
                    conf = float(keypoints[kpt_id][2])
                    if conf > 0.5:  # Confidence threshold
                        cv2.circle(frame1, (x, y), 3, (0,0,255), -1)
                        cv2.putText(frame1, str(kpt_id), (x+5, y+5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        frame1_keypoints.append([x, y])
                    else:
                        frame1_keypoints.append([-1, -1])
                else:
                    frame1_keypoints.append([-1, -1])
        else:
            frame1_keypoints = [[-1, -1]] * len(pose_keypoints)
            
        kpts_cam1.append(frame1_keypoints)

        # Calculate 3D positions using DLT
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((12, 3))
        kpts_3d.append(frame_p3ds)

        # Display the processed frames
        cv2.imshow('cam1', frame1)
        cv2.imshow('cam0', frame0)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    for cap in caps:
        cap.release()

    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)

if __name__ == '__main__':
    input_stream1 = 'media/cam0_test.mp4'
    input_stream2 = 'media/cam1_test.mp4'

    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_cam0, kpts_cam1, kpts_3d = run_yolo(input_stream1, input_stream2, P0, P1)

    write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('kpts_3d.dat', kpts_3d)