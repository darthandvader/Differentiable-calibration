import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch as th

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked coordinates
        params["keypoints"].append((x, y))
        # Display the point on the image
        cv2.circle(params["image"], (x, y), 3, (0, 1, 0), -1)
        cv2.imshow("Reference Image", params["image"])


def get_reference_keypoints(image, num_keypoints=2):
    # Load the image
    # image = cv2.imread(ref_img_path)
    # clone = image.copy()
    image = image.cpu().numpy()

    params = {"image": image, "keypoints": []}

    # Set up the mouse callback
    cv2.namedWindow("Reference Image")
    cv2.setMouseCallback("Reference Image", click_event, params)

    print(f"Please click {num_keypoints} points on the reference image.")

    while True:
        cv2.imshow("Reference Image", params["image"])
        key = cv2.waitKey(1) & 0xFF

        # Break when the required number of keypoints are collected
        if len(params["keypoints"]) >= num_keypoints:
            break

        # Exit on 'q' key press
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return params["keypoints"]


def get_reference_keypoints_auto(ref_img_path, num_keypoints=2):
    cv_img = cv2.imread(ref_img_path)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ref_img = cv2.inRange(cv_img, np.ones(3) * 128, np.ones(3) * 255) 
    binary_mask = ref_img.astype(np.uint8)
    region_mask = (binary_mask > 0).astype(np.uint8)
    max_corners = num_keypoints        # Maximum number of corners to find
    quality_level = 0.1     # Minimum quality of corners (lower means more corners)
    min_distance = 5         # Minimum distance between detected corners
    block_size = 15          # Size of the neighborhood considered for corner detection
    corners = cv2.goodFeaturesToTrack(
        binary_mask,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=region_mask
    )

    output_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Draw the corners
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(output_image, (int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-1)  # Red circles for corners

    # # Visualize the result
    # plt.figure(figsize=(8, 8))      
    # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    # plt.title("Corners of the Mask")
    # plt.axis('off')
    # plt.pause(2)  # Pause for 4 seconds
    # plt.close('all')

    ref_keypoints = corners
    print(f"detected keypoint shape: {ref_keypoints.shape}")  # [2, 1, 2] will squeeze 
     
    return ref_keypoints.squeeze(1).tolist()  # Convert to list of tuples





if __name__ == "__main__":
    # Example usage
    ref_img_path = "data/consistency_evaluation/easy/0/00026.jpg"
    keypoints = get_reference_keypoints_auto(ref_img_path)

