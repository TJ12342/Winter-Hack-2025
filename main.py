import cv2
import numpy as np

def detect_laser_pointer():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Set webcam parameters (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for colors
        # Red
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Green
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Blue
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])

        # Create masks
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Combine all masks
        combined_mask = mask_red + mask_green + mask_blue

        # Use morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 1:  # Ignore noise
                # Draw a circle to mark the laser point
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Laser Pointer Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_laser_pointer()