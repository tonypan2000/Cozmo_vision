import cozmo
from cozmo.util import degrees, Pose


def cozmo_program(robot: cozmo.robot.Robot):
    """ global coordinates from Cozmo SDK """
    # trajectory = [Pose(400, 0, 0, angle_z=degrees(90)), Pose(400, 250, 0, angle_z=degrees(180)),
    # Pose(0, 250, 0, angle_z=degrees(270)), Pose(0, 0, 0, angle_z=degrees(0))]
    # # goes around a block four times
    # for i in range(4):
    #     for coord in trajectory:
    #         robot.go_to_pose(coord).wait_for_completed()

    ''' relative motion '''
    trajectory = [Pose(400, 0, 0, angle_z=degrees(90)), Pose(250, 0, 0, angle_z=degrees(90))]
    # goes around a block four times
    for i in range(8):
        for coord in trajectory:
            robot.go_to_pose(coord, relative_to_robot=True).wait_for_completed()


cozmo.run_program(cozmo_program)