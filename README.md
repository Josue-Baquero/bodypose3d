**Real time 3D body pose estimation using MediaPipe or YOLO11**

This is a demo on how to obtain 3D coordinates of body keypoints using either MediaPipe or YOLO11 with two calibrated cameras. Two cameras are required as there is no way to obtain global 3D coordinates from a single camera. For camera calibration, my package on github [stereo calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate), my blog post on how to stereo calibrate two cameras: [link](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html). Alternatively, follow the camera calibration at Opencv documentations: [link](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html).

![input1](media/cam0_kpts.gif "input1") ![input2](media/cam1_kpts.gif "input2") 
![output](media/pose2.gif "output")

**Requirements**
```
For MediaPipe implementation:
- Mediapipe
- Python3.8
- OpenCV
- matplotlib

For YOLO11 implementation:
- ultralytics
- Python3.8
- OpenCV
- matplotlib
```

**Usage: Getting real time 3D coordinates**

The code provides two implementations for pose estimation:

1. **MediaPipe Implementation**
```bash
python bodypose3d.py  # For sample videos
python bodypose3d.py 0 1  # For webcams with IDs 0 and 1
```

2. **YOLO11 Implementation**
```bash
python bodypose3d_YOLO11.py  # For sample videos
python bodypose3d_YOLO11.py 0 1  # For webcams with IDs 0 and 1
```

Make sure the corresponding camera parameters are updated for your cameras in `camera_parameters` folder. The cameras were calibrated to 720x720p. The code crops the input image to this size. If your cameras are calibrated to a different resolution, make sure to change the code to your camera calibration.

The 3D coordinate in each video frame is recorded in `frame_p3ds` parameter. Use this for real time application. **Warning**: The code saves keypoints for all previous frames. If you run the code for long periods, then you will run out of memory. To fix this, remove append calls to: `kpts_3d, kpts_cam0, kpts_cam1`. When you press the ESC key, body keypoints detection will stop and three files will be saved to disk.

**Keypoint Identifiers**

The two implementations use different keypoint indexing systems:

1. **MediaPipe Implementation** (as shown in the original image):
![output](media/keypoints_ids.png "keypoint_ids")

2. **YOLO11 Implementation** keypoints (17 points):
- 0: Nose
- 1: Left Eye
- 2: Right Eye
- 3: Left Ear
- 4: Right Ear
- 5: Left Shoulder
- 6: Right Shoulder
- 7: Left Elbow
- 8: Right Elbow
- 9: Left Wrist
- 10: Right Wrist
- 11: Left Hip
- 12: Right Hip
- 13: Left Knee
- 14: Right Knee
- 15: Left Ankle
- 16: Right Ankle

Note that the YOLO11 implementation in this code uses a subset of these points to match the MediaPipe implementation's structure for compatibility with the visualization tools.

**Viewing 3D coordinates**

To view the recorded 3D coordinates, use either:
```bash
python show_3d_pose.py  # For MediaPipe output
python show_3d_pose_YOLO11.py  # For YOLO11 output
```

**Key Differences between Implementations**

- YOLO11 typically provides more robust pose detection in challenging conditions
- The YOLO11 implementation includes confidence scores for each keypoint
- The YOLO11 version has a more complete set of facial keypoints available
- Both implementations are compatible with the same camera calibration setup

Choose the implementation that best suits your needs based on these factors and your specific use case.
