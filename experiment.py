from enum import Enum
from utils import detect4Circles, gaussianKernel
import numpy as np
import logging

class Method(Enum):
    JACOBIAN_PINV = 1
    JACOBIAN_SVD = 2
    JACOBIAN_TRANSPOSE = 3
    KF = 4

class ExperimentStatus(Enum):
    SUCCESS = 0
    FAIL = 1

class Experiment:
    def __init__(self, q_start: list, desired_pos: list, noise_prof: object, t_s: float, t_max: float, gain: float, robot: object, method: enumerate, logger: object = None, **method_params) -> None:
       
        self.q_start = q_start
        self.desired_pos = desired_pos
        self.robot = robot
        self.noise_prof = noise_prof
        self.t_s = t_s
        self.t_max = t_max
        self.gain = gain
        self.method = method
        
        if "method_params" in method_params:
            method_params = method_params["method_params"]

        if self.method == Method.KF:
            
            self.initial_guess = method_params['initial_guess']
            self.estimate_Jinv = method_params['estimate_Jinv']

        self.logger = logging.getLogger(__name__)
        if logger is not None:
            self.logger.setLevel(logger.level)
            for handler in logger.handlers:
                self.logger.addHandler(handler)

    def run(self) -> list:
        experiment_status = ExperimentStatus.SUCCESS
    
        self.robot.start(self.q_start)

        noise = np.zeros(len(self.desired_pos))

        error_log = np.zeros((int(self.t_max/self.t_s), len(self.desired_pos)))
        q_log = np.zeros((int(self.t_max/self.t_s), 6))
        camera_log = np.zeros((int(self.t_max/self.t_s), 6))
        t_log = np.zeros(int(self.t_max/self.t_s))
        desired_pos_log = np.zeros((int(self.t_max/self.t_s), len(self.desired_pos)))
        noise_log = np.zeros((int(self.t_max/self.t_s), len(self.desired_pos)))

        # Main loop
        scale_matrix_t = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0],[0,0,0]])
        k = 0

        '''
        if self.method == Method.KF:
            X = np.zeros((3 * 6, 1)) # For position, if orientation too, should chance the number 3
            Z = np.zeros((3, 1))
            H = np.zeros((3, 3 * 6))
            P = np.eye(3 * 6)
            K = np.zeros((3 * 6, 3))
            Q = np.eye(3 * 6)
            R = np.eye(3)
        '''

        while (t := self.robot.sim.getSimulationTime()) < self.t_max:
            
            X_m = self.robot.computePose(recalculate_fkine=True)
            error = self.desired_pos - X_m[0:3]

            # Calculating / Estimating jacobian
            J = self.robot.jacobian_position()
            
            dq = np.zeros((6,1))

            try:
                if self.method == Method.JACOBIAN_PINV:
                    dq = self.gain * np.linalg.pinv(J) @ error.reshape(3, 1)
                elif self.method == Method.JACOBIAN_SVD:
                    u, _, vt = np.linalg.svd(J)
                    dq = self.gain * (vt.T @ scale_matrix_t @ u.T) @ error.reshape(3, 1)
                elif self.method == Method.JACOBIAN_TRANSPOSE:
                    dq = self.gain * J.T @ error.reshape(3, 1)
            except:
                experiment_status = ExperimentStatus.FAIL
                self.logger.error('Experiment failed')
                break

            # Invkine
            new_q = self.robot.getJointsPos() + dq.ravel() * self.t_s

            # Logging
            q_log[k] = self.robot.getJointsPos()
            camera_log[k] = self.robot.computePose()
            desired_pos_log[k] = self.desired_pos
            error_log[k] = error
            noise_log[k] = noise
            t_log[k] = t
            k += 1
            self.logger.debug('time: ' + str(t) + '; error: ' + str(np.linalg.norm(error)))

            # Send theta command to robot
            self.robot.setJointsPos(new_q)

            # next_step
            self.robot.step()
    
        t_log = np.delete(t_log, [i for i in range(k, len(t_log))], axis=0)
        error_log = np.delete(error_log, [i for i in range(k, len(error_log))], axis=0)
        q_log = np.delete(q_log, [i for i in range(k, len(q_log))], axis=0)
        desired_pos_log = np.delete(desired_pos_log, [i for i in range(k, len(desired_pos_log))], axis=0)
        camera_log = np.delete(camera_log, [i for i in range(k, len(camera_log))], axis=0)
        noise_log = np.delete(noise_log, [i for i in range(k, len(noise_log))], axis=0)
        
        self.robot.stop()

        if experiment_status == ExperimentStatus.SUCCESS:
            self.logger.info("Experiment success")

        return experiment_status, t_log, error_log, q_log, desired_pos_log, camera_log, noise_log