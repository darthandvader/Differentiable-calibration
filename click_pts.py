import cv2


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the clicked coordinates
        params['keypoints'].append((x, y))
        # Display the point on the image
        cv2.circle(params['image'], (x, y), 3, (0, 1, 0), -1)
        cv2.imshow('Reference Image', params['image'])

def get_reference_keypoints(image, num_keypoints=2):
    # Load the image
    # image = cv2.imread(ref_img_path)
    # clone = image.copy()
    image= image.cpu().numpy()
        
    params = {'image': image, 'keypoints': []}

    # Set up the mouse callback
    cv2.namedWindow('Reference Image')
    cv2.setMouseCallback('Reference Image', click_event, params)

    print(f"Please click {num_keypoints} points on the reference image.")

    while True:
        cv2.imshow('Reference Image', params['image'])
        key = cv2.waitKey(1) & 0xFF

        # Break when the required number of keypoints are collected
        if len(params['keypoints']) >= num_keypoints:
            break

        # Exit on 'q' key press
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return params['keypoints']

    
if __name__ == '__main__': 
    ref_img_path = "data/extra_set1/00075.jpg"
    
    keypoints = get_reference_keypoints(ref_img_path)
    print(keypoints)
    