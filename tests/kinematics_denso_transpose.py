##
#   @author glauberrleite
##
import numpy as np

import sys
sys.path.append("..")

from denso_simulation import DensoVP6242Simulation

GAIN = 0.5
T_MAX = 50

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

itae = 0
t_old = 0

while (t := robot.sim.getSimulationTime()) < T_MAX:

    dt = t - t_old

    # Compute error
    X_m = robot.computePose(recalculate_fkine=True)
#    error_quat = X_d - X_m
    error = X_d[0:3] - X_m[0:3]

#   error = np.zeros(6)
#    error[0:3] = error_quat[0:3]
#   if t > 35:
#        error[3], error[4], error[5] = quat2taitbryan(error_quat[3:7])

    J = robot.jacobian_position(recalculate_fkine=True)

    # Inverse Kinematics Control Law
    dq = GAIN * J.T @ error.reshape(3, 1)

    new_q = robot.getJointsPos() + dq.ravel() * dt

    k += 1

    print(t)
    print(X_m)
    print(X_d)
    print(error)
    print('-------')

    itae += t * np.linalg.norm(error) * dt
    t_old = t

    # Send theta command to robot
    robot.setJointsPos(new_q)

    # next_step
    robot.step()

print(itae)