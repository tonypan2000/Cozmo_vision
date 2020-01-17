import sys
import cozmo
import cv2
import numpy as np
from apriltags3 import Detector

from cozmo.util import degrees

try:
    from PIL import ImageDraw, ImageFont, Image, ImageTk
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')


def cozmo_program(robot: cozmo.robot.Robot):
    # Move lift down and tilt the head up
    robot.move_lift(-3)
    robot.set_head_angle(degrees(0)).wait_for_completed()

    # intrinsics from calibration.cfg
    cameraMatrix = np.array([[288.15418237, 0, 197.31722863],
                             [0, 285.12686892, 120.45748409],
                             [0, 0, 1]])
    # 3D coordinates of the center of AprilTags in the arm frame in meters.
                    #           x         y           x (meters in Cozmo camera coordinate frame)
    objectPoints = np.array([[0.15, 0.0254 / 2, -.057 + 0.0254 / 2],
                             [0.15 + 0.0254, -0.0254 / 2, -.057 + 0.0254 / 2],
                             [0.15, 0.0254 / 2, -.057 + (3 * 0.0254 / 2)],
                             [0.15 + 0.0254, -0.0254 / 2, -.057 + (3 * .0254 / 2)]])

    detector = Detector("tagStandard41h12", quad_decimate=2.0, quad_sigma=1.0, debug=False)

    # create main window
    cv2.namedWindow("camera", 1)

    while True:
        ch = 0xFF & cv2.waitKey(10)
        # convert Bayer GB to RGB for display
        image = robot.world.latest_image.raw_image
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert Bayer BG to Grayscale for corner detections
        tags = detector.detect(gray, estimate_tag_pose=True, camera_params=cameraMatrix, tag_size=0.0127)
        # visualize the detection
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(gray, tuple(tag.corners[idx - 1, :].astype(int)),
                         tuple(tag.corners[idx, :].astype(int)), (255, 0, 0))

            # label the id of AprilTag on the image.
            cv2.putText(gray, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 0, 0))
        # continue until ESC
        if ch == 27:
            break

    # Use the center of the tags as image points. Make sure they correspond to the 3D points.
    imagePoints = np.array([tag.center for tag in tags])
    assert (len(tags) == 4)
    print("Image Points: ", imagePoints)
    success, rvec, tvec = cv2.solvePnP(objectPoints, np.array(imagePoints), cameraMatrix, None)
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    affine_transformation = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], tvec[0]],
                         [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], tvec[1]],
                         [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], tvec[2]],
                         [0.0, 0.0, 0.0, 1.0]], dtype='float')
    # homogeneous matrix from camera coordinates to robot coordinates
    extrinsic = np.linalg.inv(affine_transformation)
    print("extrinsic: ", extrinsic)

    f = open('extrinsics.cfg', 'w')
    f.write("extrinsics matrix:\r\n")
    f.write(str(extrinsic))
    f.close()

    cv2.destroyAllWindows()


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
