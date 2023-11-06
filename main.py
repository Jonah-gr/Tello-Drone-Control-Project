from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import mediapipe as mp
import os

# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number also results in input lag, as input information is processed once per frame.
FPS = 120



class TelloInterface(object):
    """ 
    Maintains the Tello display and moves it through the keyboard keys.
    Press escape key to quit.
    The controls are:
        - T: Takeoff
        - L: Land
        - Arrow keys: Forward, backward, left and right.
        - A and D: Counter clockwise and clockwise rotations (yaw)
        - W and S: Up and down.
        - Z,G,H,J: flip (forward, back, left, right) NOT IMPLEMENTED YET!
    """
    def __init__(self, face_detection_on = False, gesture_control_on = False, record_video = False):
        """
        Initialize the TelloController object.

        Args:
            face_detection_on (bool): Enable or disable face detection.
            gesture_control_on (bool): Enable or disable gesture control.
            record_video (bool): Enable or disable video recording.
        """
        # Init pygame
        pygame.init()

        # Attributes
        self.face_detection_on = face_detection_on
        self.gesture_control_on = gesture_control_on
        self.record_video = record_video

        # Models for face recognition, gesture control and video recording
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.video_writer = None

        # Create pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([720, 480])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)


    def run(self):
        """
        Run the main control loop for the Tello drone.

        This method connects to the Tello, sets up the video stream,
        handles user events, displays the video stream in a Pygame window,
        and updates the drone's state based on user input.

        This method continues running until the user initiates a stop signal.
        """
        # Connect to the Tello drone and set the speed
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # Turn off and then turn on the video stream from the drone
        self.tello.streamoff()
        self.tello.streamon()

        # Get the frame reader from the drone
        frame_read = self.tello.get_frame_read()

        should_stop = False

        while not should_stop:
        # Check for pygame events
            for event in pygame.event.get():
                # Update the display
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                # Check if the user closes the window
                elif event.type == pygame.QUIT:
                    should_stop = True
                # Check for keydown events
                elif event.type == KEYDOWN:
                    print(event.key)
                    # Check if the Escape key is pressed to stop the program
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        # Call the keydown method to handle the key press
                        self.keydown(event.key)
                        self.update()
                # Check for keyup events
                elif event.type == KEYUP:
                    print(event.key)
                    # Call the keyup method to handle the key release
                    self.keyup(event.key)
                    self.update()

            # Clear the screen
            self.screen.fill([0, 0, 0])

            # Read the frame from the frame reader
            frame = frame_read.frame

            # Display the battery percentage on the frame
            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, frame.shape[0] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Perform face detection if enabled 
            if self.face_detection_on:
                    self.face_detection(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform gesture control if enabled
            if self.gesture_control_on:
                self.gesture_control(frame)

            # Record video and save video stream if enabled
            if self.record_video:  
                self.video_recorder(frame)

            # Rotate and flip the frame
            frame = np.rot90(frame)
            frame = np.flipud(frame)     
            
            # Convert the frame to a Pygame surface and scale it
            frame = pygame.surfarray.make_surface(frame)
            frame = pygame.transform.scale(frame, (720, 480))

            # Blit the frame onto the screen and update the display
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call the end method to deallocate resources before finishing
        self.tello.streamoff()
        self.tello.end()
        if self.video_writer is not None:
            self.video_writer.release()
        pygame.quit()
  

    def face_detection(self, frame):
        """
        Perform face detection on the given frame.

        Args:
            frame: The frame to perform face detection on.
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize the grayscale frame to a smaller size
        img = cv2.resize(gray, (160, 120))
        
        # Perform face detection using the self.face_cascade classifier
        # Adjust the scaleFactor, minNeighbors, and minSize parameters as needed
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5))
        
        # Iterate over the detected faces and draw rectangles on the frame
        for (x, y, w, h) in list(map(lambda x: tuple(value * 6 for value in x), faces)):
            # Scale the coordinates back to the original frame size
            # Multiply each coordinate by 6 to match the scaling factor used earlier
            # Draw a green rectangle around each face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    def gesture_control(self, frame):
        """
        Perform gesture control based on hand tracking using the MediaPipe Hands module.

        Args:
            frame: The frame to process and perform gesture control on.
        """
        # Initialize the MediaPipe Hands module for hand tracking
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
            # Resize the frame to a smaller size
            resized_frame = cv2.resize(frame, (120, 90))
            # resized_frame = frame
            
            # Process the resized frame using the hand tracking module
            results = hands.process(resized_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Draw landmarks and connections on the frame
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Get the coordinates of the index finger tip
                    finger_coords = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    curr_x, curr_y = int(finger_coords.x * frame.shape[1]), int(finger_coords.y * frame.shape[0])

                    # Calculate the horizontal and vertical movement
                    dx = curr_x - frame.shape[1] / 2
                    dy = curr_y - frame.shape[0] / 2

                    # Calculate the hand size
                    hand_size = hand_landmarks.landmark[0].x - hand_landmarks.landmark[9].x
                    hand_size *= frame.shape[1]
                   
                    # Perform gesture control based on the hand movements and size
                    if dx > 100:
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_RIGHT))
                        pygame.event.post(pygame.event.Event(KEYUP, key=K_RIGHT))
                        if dx > 200:
                            pygame.event.post(pygame.event.Event(KEYDOWN, key=K_d))
                            pygame.event.post(pygame.event.Event(KEYUP, key=K_d))
                    elif dx < -100:
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_LEFT))
                        pygame.event.post(pygame.event.Event(KEYUP, key=K_LEFT))
                        if dx < -200:
                            pygame.event.post(pygame.event.Event(KEYDOWN, key=K_a))
                            pygame.event.post(pygame.event.Event(KEYUP, key=K_a))
                    if dy > 100:
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_s))
                        pygame.event.post(pygame.event.Event(KEYUP, key=K_s))
                    elif dy < -100:
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_w))
                        pygame.event.post(pygame.event.Event(KEYUP, key=K_w))
                    if hand_size < -50:
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_UP))
                        pygame.event.post(pygame.event.Event(KEYUP, key=K_UP))
                    elif hand_size > 100:
                        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_DOWN))
                        pygame.event.post(pygame.event.Event(KEYUP, key=K_DOWN))
                    

    def video_recorder(self, frame):
        """
        Record video frames into a video file.

        Args:
            frame: The frame to be recorded.

        Note:
            This function initializes the video writer if it's not already created,
            and writes the frames to the video file.
        """
        if self.video_writer == None:
            # Get the dimensions of the frame
            width, height, _ = self.tello.get_frame_read().frame.shape
            
            # Determine the path for storing video files
            path = os.path.dirname(os.path.join(os.path.abspath(__file__), 'videos'))

            # Check if the directory at the specified path exists
            if not os.path.exists(path):
                # If the directory does not exist, create it
                os.mkdir(path)
            
            # Create a unique filename using the current date and time
            name = time.strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
            
            # Define the video codec
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            
            # Create the VideoWriter object with the specified filename, codec, frame rate, and dimensions
            self.video_writer = cv2.VideoWriter(os.path.join(path, name), fourcc, 30, (height, width))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the video file
        self.video_writer.write(frame)
        
   
    def keydown(self, key):
        """ 
        Update velocities based on key pressed Arguments:

        Args:
            key: pygame key that was just pressed
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S


    def keyup(self, key):
        """ 
        Update velocities based on key released Arguments:
        
        Args: 
            key: pygame key that was just released
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            not self.tello.land()
            self.send_rc_control = False
        # elif key == pygame.K_z:
        #     self.tello.flip_forward()
        # elif key == pygame.K_h:
        #     self.tello.flip_back()
        # elif key == pygame.K_g:
        #     self.tello.flip_left()
        # elif key == pygame.K_j:
        #     self.tello.flip_right()


    def update(self):
        """ 
        Update routine. Send velocities to Tello.
        """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)



def main():
    interface = TelloInterface(gesture_control_on=True, face_detection_on=True, record_video=True)

    # run interface
    interface.run()


if __name__ == '__main__':
    main()
