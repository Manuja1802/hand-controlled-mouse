import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Get the screen size
screen_width, screen_height = pyautogui.size()

video = cv2.VideoCapture(0)

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# To store previous Y position for scrolling calculation
previous_y_position = None

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while video.isOpened():
        _, frame = video.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the index, middle, and thumb finger tip landmarks
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Scale the coordinates to the screen size
                index_finger_x = int(index_finger_landmark.x * screen_width)
                index_finger_y = int(index_finger_landmark.y * screen_height)

                pyautogui.moveTo(index_finger_x, index_finger_y)

                # Get the positions of the index, middle, and thumb fingers in image coordinates
                index_finger_x_image = int(index_finger_landmark.x * image_width)
                index_finger_y_image = int(index_finger_landmark.y * image_height)
                middle_finger_x_image = int(middle_finger_landmark.x * image_width)
                middle_finger_y_image = int(middle_finger_landmark.y * image_height)
                thumb_x_image = int(thumb_tip_landmark.x * image_width)
                thumb_y_image = int(thumb_tip_landmark.y * image_height)

                # --------- Clicking (Index + Middle Finger) ---------

                # Calculate the distance between index and middle fingers (for clicking)
                index_middle_distance = calculate_distance(index_finger_x_image, index_finger_y_image, middle_finger_x_image, middle_finger_y_image)

                # Draw a yellow line between the index and middle fingers to visualize the distance
                cv2.line(image, (index_finger_x_image, index_finger_y_image), (middle_finger_x_image, middle_finger_y_image), (0, 255, 255), 2)

                # Define a threshold distance for "click"
                click_threshold = 40  # Adjust based on camera resolution and distance between fingers

                # Simulate a click if the distance is below the threshold
                if index_middle_distance < click_threshold:
                    print("Clicking!")
                    pyautogui.click()  # Simulate a click when index and middle fingers are close

                # --------- Scrolling Based on Thumb and Index Finger ---------

                # Calculate the distance between the thumb and index finger (for scrolling)
                thumb_index_distance = calculate_distance(thumb_x_image, thumb_y_image, index_finger_x_image, index_finger_y_image)

                # Draw a blue line between the thumb and index finger to visualize the distance
                cv2.line(image, (thumb_x_image, thumb_y_image), (index_finger_x_image, index_finger_y_image), (255, 0, 0), 2)

                # Define a threshold distance for detecting when the thumb and index fingers are joined
                scroll_threshold = 40  # Adjust this threshold for sensitivity

                # Check if thumb and index finger are close (joined) for scrolling
                if thumb_index_distance < scroll_threshold:
                    # If the fingers are joined, check the vertical movement for scrolling
                    if previous_y_position is not None:
                        # Calculate vertical movement of the joined fingers
                        y_movement = previous_y_position - index_finger_y_image

                        if y_movement > 5:  # Moving upward
                            print("Scrolling Down")
                            pyautogui.scroll(-45)  # Increased scroll down speed (from -5 to -45)

                        elif y_movement < -5:  # Moving downward
                            print("Scrolling Up")
                            pyautogui.scroll(45)  # Increased scroll up speed (from 5 to 45)

                    # Update the previous Y position for the next frame
                    previous_y_position = index_finger_y_image
                else:
                    # Reset the previous position if fingers are not joined
                    previous_y_position = None

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
