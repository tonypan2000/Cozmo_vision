import sys
import cv2
import numpy as np
import cozmo

from cozmo.util import degrees

try:
    from PIL import ImageDraw, ImageFont, Image, ImageTk
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')

font = cv2.FONT_HERSHEY_SIMPLEX

h = 480
w = 640
# size of each checker square [mm]
square_size = 25.4
# pattern of corners on checker board
pattern_size = (8, 6)

# builds array of reference corner locations
pattern_points = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
pattern_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
pattern_points *= square_size

# stores the locations of corners on the checkerboard
obj_points = []

# stores the pixel locations of corners for all frames
img_points = []

# termination criteria for finding fit
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def cozmo_program(robot: cozmo.robot.Robot):
    # Move lift down and tilt the head up
    robot.move_lift(-3)
    robot.set_head_angle(degrees(0)).wait_for_completed()

    # create main window
    cv2.namedWindow("camera", 1)

    frame_count = 0
    while True:
        ch = 0xFF & cv2.waitKey(10)
        if True:
            # convert Bayer GB to RGB for display
            image = robot.world.latest_image.raw_image
            img = np.array(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # convert Bayer BG to Grayscale for corner detections
            frames_str = "Frames:" + str(frame_count)
            cv2.putText(img, frames_str, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # space key
            if ch == 32:
                print("searching... ")
                # find location of corners in image, this is really slow if no corners are seen.
                found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
                if found:
                    # find sub pixel estimate for corner location
                    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                    # add detected corners to RGB image
                    frame_count += 1
                    img_points.append(corners.reshape(-1, 2))
                    obj_points.append(pattern_points)
                    cv2.drawChessboardCorners(img, pattern_size, corners, found)
                    cv2.imshow('camera', img)
            else:
                cv2.imshow('camera', img)
        # continue until ESC
        if ch == 27:
            break

    # Perform actual calibration
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print("Performing calibration with", frame_count, "frames")
    print("RMS Error:", rms)
    print("camera matrix:\r\n", camera_matrix)
    print("distortion coefficients:\r\n", dist_coefs.ravel())

    f = open('calibration.cfg', 'w')
    f.write("intrinsic matrix:\r\n")
    f.write(str(camera_matrix))
    f.write("\r\ndistortion coefficients:\r\n")
    f.write(str(dist_coefs.ravel()))
    f.close()

    cv2.destroyAllWindows()


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
