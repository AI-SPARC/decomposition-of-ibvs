##
#   @author glauberrleite
##
import numpy as np

import sys
sys.path.append("..")

from denso_simulation import DensoVP6242Simulation
from utils import quat2taitbryan

from noise import NoiseProfiler, NoiseType

TS = 0.05
GAIN = 0.1
T_MAX = 50

PERSPECTIVE_ANGLE = 0.65

print("Instantiating robot")
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Desired starting configuration
#q = np.array([0.0, -np.pi/8, np.pi/8, 0.0, -np.pi/2, 0.0]) # Desired starting configuration
robot = DensoVP6242Simulation()
robot.start(q)

# Waiting robot to arrive at starting location
print("Moving robot to starting position")
while (t := robot.sim.getSimulationTime()) < 3:
    robot.step()

#X_d = np.array([0.25, 0.0, 0.45, np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]) # Center
X_d = np.array([0.25, 0.0, 0.45, -np.pi, 0, 0]) # Center
X_m = robot.computePose(recalculate_fkine=True)
print(X_m)
print(X_d)
print('-------')

input()

k = 0

while (t := robot.sim.getSimulationTime()) < T_MAX:

    # Compute error
    X_m = robot.computePose(recalculate_fkine=True)
#    error_quat = X_d - X_m
    error = X_d - X_m

#   error = np.zeros(6)
#    error[0:3] = error_quat[0:3]
#   if t > 35:
#        error[3], error[4], error[5] = quat2taitbryan(error_quat[3:7])

    J = robot.jacobian(recalculate_fkine=True)

    # Inverse Kinematics Control Law
    dq = GAIN * np.linalg.pinv(J) @ error.reshape(6, 1)

    new_q = robot.getJointsPos() + dq.ravel() * TS

    k += 1

    print(t)
    print(X_m)
    print(X_d)
    print(error)
    print('-------')

    # Send theta command to robot
    robot.setJointsPos(new_q)

    # next_step
    robot.step()