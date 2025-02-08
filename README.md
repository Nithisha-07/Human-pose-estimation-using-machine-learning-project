# Human-pose-estimation-using-machine-learning-project
import torch
import torchvision
import cv2  # For image processing
import numpy as np

# Load a pre-trained pose estimation model (e.g., from torchvision's keypoint detection models)
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the COCO keypoints (you might need to adjust this based on the model)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Function to perform pose estimation on an image
def estimate_pose(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB (PyTorch uses RGB)

    # Preprocess the image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # Convert to tensor
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
    ])

    img_tensor = transform(img).unsqueeze(0) # Add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Process the predictions
    keypoints = predictions[0]['keypoints']
    scores = predictions[0]['keypoints_scores']

    # Draw the pose on the image
    img_with_pose = img.copy()
    for kps, scs in zip(keypoints, scores):
      for kp, sc in zip(kps, scs):
          if sc > 0.5: # Example score threshold
              x, y = int(kp[0]), int(kp[1])
              cv2.circle(img_with_pose, (x, y), 5, (0, 255, 0), -1) # Draw a circle for each keypoint

    img_with_pose = cv2.cvtColor(img_with_pose, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV
    return img_with_pose, keypoints, scores


# Example usage
image_path = "path/to/your/image.jpg"  # Replace with the actual path
img_with_pose, keypoints, scores = estimate_pose(image_path)

if img_with_pose is not None:
  cv2.imshow("Pose Estimation", img_with_pose)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  print("Keypoints:", keypoints)
  print("Scores:", scores)
else:
  print("Error processing image or no pose detected.")
  
