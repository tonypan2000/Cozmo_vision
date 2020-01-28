import sys
import cozmo
import cv2
import numpy as np
from apriltags3 import Detector

from cozmo.util import degrees, distance_mm, speed_mmps

try:
    from PIL import ImageDraw, ImageFont, Image, ImageTk
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')


def cozmo_program(robot: cozmo.robot.Robot):
    # Move lift down and tilt the head up
    robot.move_lift(-3)
    robot.set_head_angle(degrees(0)).wait_for_completed()
    # intrinsics from calibration.cfg
    cameraParam = [288.15418237, 285.12686892, 197.31722863, 120.45748409]  # [fx, fy, cx, cy]
    cameraMatrix = np.array([[288.15418237, 0, 197.31722863], [0, 285.12686892, 120.45748409], [0, 0, 1]])
    extrinsic = np.array([[-9.97469913e-02, -5.61812756e-02, 9.93425489e-01, -2.40474116e-02],
                          [-9.94999768e-01, 5.15631732e-04, -9.98758996e-02, -4.49931548e-03],
                          [5.09891373e-03, -9.98420452e-01, -5.59517889e-02, 3.75532293e-02],
                          [0.,         0.,         0.,          1.        ]])

    detector = Detector("tagStandard41h12", quad_decimate=2.0, quad_sigma=1.0, debug=False)

    # create main window
    cv2.namedWindow("AprilTag Detection", 1)

    while True:
        robot.set_head_angle(degrees(0)).wait_for_completed()
        ch = 0xFF & cv2.waitKey(10)
        # convert Bayer GB to RGB for display
        image = robot.world.latest_image.raw_image
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert Bayer BG to Grayscale for corner detections
        tags = detector.detect(gray, estimate_tag_pose=True, camera_params=cameraParam, tag_size=0.0127)

        # visualize the detection
        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(gray, tuple(tag.corners[idx - 1, :].astype(int)),
                         tuple(tag.corners[idx, :].astype(int)), (255, 0, 0))

            rot_mat = np.array([[tag.pose_R[0][0], tag.pose_R[0][1], tag.pose_R[0][2], tag.pose_t[0]],
                             [tag.pose_R[1][0], tag.pose_R[1][1], tag.pose_R[1][2], tag.pose_t[1]],
                             [tag.pose_R[2][0], tag.pose_R[2][1], tag.pose_R[2][2], tag.pose_t[2]],
                             [0.0, 0.0, 0.0, 1.0]], dtype='float')
            tag_pose = np.matmul(rot_mat, np.array([0, 0, 0.0125, 1]))
            pose_in_Cozmo = np.matmul(extrinsic, tag_pose)
            # label the id of AprilTag on the image.
            cv2.putText(gray, str(pose_in_Cozmo),
                        org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.3,
                        color=(255, 0, 0))
        cv2.imshow('AprilTags', gray)
        if len(tags) > 0:
            # first turn towards target
            angle = np.rad2deg(np.arctan2(pose_in_Cozmo[1], pose_in_Cozmo[0]))
            robot.turn_in_place(degrees(angle)).wait_for_completed()
            # then drive straight
            distance = np.sqrt(pose_in_Cozmo[0] ** 2 + pose_in_Cozmo[1] ** 2) * 1000
            robot.drive_straight(distance_mm(distance), speed_mmps(50)).wait_for_completed()

        # continue until ESC
        if ch == 27:
            break


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
