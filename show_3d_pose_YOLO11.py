import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import DLT

# YOLO11 keypoint indices
pose_keypoints = np.array([10, 8, 6, 5, 7, 9, 12, 11, 14, 13, 16, 15])  # Ordered as in bodypose3d.py

def read_keypoints(filename):
    """Read keypoints from file"""
    with open(filename, 'r') as fin:
        kpts = []
        for line in fin:
            if line == '': 
                break
            
            values = [float(s) for s in line.split()]
            points = np.reshape(values, (len(pose_keypoints), -1))
            kpts.append(points)
            
    return np.array(kpts)

def visualize_3d(p3ds):
    """Visualize 3D poses with correct YOLO11 keypoint connections"""
    # Define body connections using the index in the pose_keypoints array
    # The indices here refer to the position in pose_keypoints, not the YOLO keypoint IDs
    torso = [
        [2, 3],  # Right shoulder to left shoulder
        [2, 6],  # Right shoulder to right hip
        [3, 7],  # Left shoulder to left hip
        [6, 7]   # Right hip to left hip
    ]
    arm_r = [
        [2, 1],  # Right shoulder to right elbow
        [1, 0]   # Right elbow to right wrist
    ]
    arm_l = [
        [3, 4],  # Left shoulder to left elbow
        [4, 5]   # Left elbow to left wrist
    ]
    leg_r = [
        [6, 8],  # Right hip to right knee
        [8, 10]  # Right knee to right ankle
    ]
    leg_l = [
        [7, 9],  # Left hip to left knee
        [9, 11]  # Left knee to left ankle
    ]
    
    body = [torso, arm_l, arm_r, leg_r, leg_l]
    colors = ['red', 'blue', 'green', 'black', 'orange']
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each frame
    for framenum, kpts3d in enumerate(p3ds):
        if framenum % 2 == 0:  # Skip every other frame for smoother visualization
            continue
            
        # Plot each body part
        for bodypart, part_color in zip(body, colors):
            for connection in bodypart:
                # Get 3D coordinates for the connection
                start = kpts3d[connection[0]]
                end = kpts3d[connection[1]]
                
                # Plot the line if both points are valid
                if -1 not in start and -1 not in end:
                    ax.plot(
                        xs=[start[0], end[0]],
                        ys=[start[1], end[1]],
                        zs=[start[2], end[2]],
                        linewidth=4,
                        color=part_color
                    )
        
        # Set plot properties
        ax.set_title('3D Pose Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set axis limits
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10, 10)
        ax.set_zlim3d(-10, 10)
        
        # Add grid
        ax.grid(True)
        
        # Pause to create animation effect
        plt.pause(0.1)
        ax.cla()  # Clear for next frame
    
    plt.close()

if __name__ == '__main__':
    try:
        # Read the 3D keypoints
        print("Loading keypoints from kpts_3d.dat...")
        p3ds = read_keypoints('kpts_3d.dat')
        
        print(f"Loaded {len(p3ds)} frames of 3D pose data")
        print("Starting visualization...")
        
        # Visualize the poses
        visualize_3d(p3ds)
        
    except FileNotFoundError:
        print("Error: Could not find kpts_3d.dat file.")
        print("Please make sure to run bodypose3d.py first to generate the keypoints file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")