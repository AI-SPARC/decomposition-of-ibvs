from experiment import Experiment, Method, ExperimentStatus
from noise import NoiseProfiler, NoiseType
from denso_simulation import DensoVP6242Simulation
import numpy as np
import pandas as pd
import time
import logging
from json import load, dump
import sys
import os

directory_path = os.path.join('./results/data', time.strftime("%d_%m_%Y_%H_%M_%S", time.localtime()))
os.mkdir(directory_path)

# Reading config
with open("config.json", "r", encoding="utf-8") as config_file:
    config = load(config_file)
    
    try:
        # Logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)

        if config["log_level"] == "DEBUG":
            ch.setLevel(logging.DEBUG)
        elif config["log_level"] == "INFO":
            ch.setLevel(logging.INFO)
        elif config["log_level"] == "WARNING":
            ch.setLevel(logging.WARNING)
        elif config["log_level"] == "ERROR":
            ch.setLevel(logging.ERROR)
        elif config["log_level"] == "CRITICAL":
            ch.setLevel(logging.CRITICAL)

        fh = logging.FileHandler(os.path.join(directory_path, 'debug.log'))
        fh.setLevel(logging.DEBUG)

        log_formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(log_formatter)
        ch.setFormatter(log_formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        # Experiments config
        logger.info("Loading experiments config")
        experiments_config = config["experiments"]

        for key, value in experiments_config.items():
            logger.info(str(key) + " : " + str(value))

        dt = experiments_config["dt"]
        t_max = experiments_config["t_max"]
        epoch = experiments_config["epoch"]
        gain = experiments_config["gain"]
        q_start = np.array(experiments_config["q_start"])
        desired_pos = np.array(experiments_config["desired_pos"])
        visualization = experiments_config["visualization"]
        change_q_start = experiments_config["change_q_start"]
        experiment_seed = experiments_config["seed"]

        # Estimator config
        logger.info("Loading estimator config")
        estimator_config = config["estimator"]
        
        for key, value in estimator_config.items():
            logger.info(str(key) + " : " + str(value))

        method_name = estimator_config["method"]
        if method_name in [e.name for e in Method]:
            method = Method[method_name]
        else: 
            logger.critical("Estimation method " + method_name + " unknown.")
            sys.exit()
        method_params = estimator_config["estimator_params"]

        # Noise config
        logger.info("Loading noise config")
        noise_config = config["noise"]

        for key, value in noise_config.items():
            logger.info(str(key) + " : " + str(value))

        noise_type_name = noise_config["type"]
        if noise_type_name in [e.name for e in NoiseType]:
            noise_type = NoiseType[noise_type_name]
        else: 
            logger.critical("Noise type " + noise_type_name + " unknown.")
            sys.exit()
        noise_params = noise_config["noise_params"]
        noise_hold = noise_config["hold"]
        noise_hold_time = noise_config["hold_time"]
        seed = noise_config["seed"]
    
    except KeyError as e:
        logger.critical("Could not load config key: " + str(e))
        sys.exit()
    
    # Saving a copy in the experiments directory
    with open(os.path.join(directory_path, "config.json"), "w", encoding="utf-8") as json_copy:
        dump(config, json_copy)

rho_list = np.linspace(0, 0.2, 12)
if noise_type == NoiseType.ALPHA_STABLE:
    rho_list = np.linspace(1, 2, 12)

#### GAUSSIAN
rho_list = [0]
#### GAUSSIAN

experiments = []

file_header = True

k = 0

experiment_success_cnt = 0
experiment_fail_cnt = 0

if change_q_start:
    random_prof = NoiseProfiler(num_features=6, noise_type=NoiseType.UNIFORM, seed=experiment_seed, logger=logger)

# Preparing experiment queue
for rho in rho_list:
    
    if noise_type == NoiseType.ALPHA_STABLE:
        noise_params["alpha"] = rho
    else:
        noise_params["rho"] = rho
    for i in range(epoch):
        # Setting q starting value
        q = q_start.copy()
        if change_q_start:
            # With these little changes, the robot camera still starts with the features in the view
            random_values = random_prof.getNoise() # Get uniform random values from 0 to 1
            q[0] = q_start[0] + 2 * (random_values[0] - 1) * (np.pi/9)
            q[1] = q_start[1] + 2 * (random_values[1] - 1) * (np.pi/9)
            q[2] = q_start[2] + 2 * (random_values[2] - 1) * (np.pi/9)
            q[3] = q_start[3] + 2 * (random_values[3] - 1) * (np.pi/9)
            q[4] = q_start[4] + 2 * (random_values[4] - 1) * (np.pi/9)
            q[5] = q_start[5] + 2 * (random_values[5] - 1) * (np.pi/9)

        # Noise generation
        noise_prof = NoiseProfiler(num_features=len(desired_pos), noise_type=noise_type, seed=seed, logger=logger, noise_hold=noise_hold, noise_hold_cnt=int(noise_hold_time/dt) , noise_params=noise_params)
        if seed is not None:
            seed = seed + 1

        robot = DensoVP6242Simulation(logger=logger, visualization=visualization)
        experiment = Experiment(q_start=q, desired_pos=desired_pos, noise_prof=noise_prof, t_s=dt, t_max=t_max, gain=gain, robot=robot, logger=logger, method=method, method_params=method_params)

        logger.info("Experiment " + str(k+1) + " of " + str(len(rho_list)*epoch))
        logger.info("Noise params: " + str(noise_params))
        
        # Running experiment
        status, t_log, error_log, q_log, desired_pos_log, camera_log, noise_log = experiment.run()

        # Saving data
        logger.debug("Saving experiment data in csv")  
        dataframe = pd.DataFrame(data={
            'experiment_id': k,
            'status': status,
            'rho': rho, # Temporary
            't': t_log,
            'q_1': q_log[:, 0],
            'q_2': q_log[:, 1],
            'q_3': q_log[:, 2],
            'q_4': q_log[:, 3],
            'q_5': q_log[:, 4],
            'q_6': q_log[:, 5],
            'camera_x': camera_log[:, 0],
            'camera_y': camera_log[:, 1],
            'camera_z': camera_log[:, 2],
            'camera_roll': camera_log[:, 3],
            'camera_pitch': camera_log[:, 4],
            'camera_yaw': camera_log[:, 5],
            'desired_pos_1': desired_pos_log[:, 0],
            'desired_pos_2': desired_pos_log[:, 1],
            'desired_pos_3': desired_pos_log[:, 2],
            'noise_1': noise_log[:, 0],
            'noise_2': noise_log[:, 1],
            'noise_3': noise_log[:, 2]
        })
    
        dataframe.to_csv(os.path.join(directory_path, 'results.csv'), mode='a', index=False, header=file_header)

        file_header = False # Removes file header for the next experiments

        k = k+1

        if status == ExperimentStatus.SUCCESS:
            experiment_success_cnt += 1
        else:
            experiment_fail_cnt += 1

logger.info("Ending experiment batch")
logger.info("Experiments status summary: " + str(experiment_fail_cnt) + " fail | " + str(experiment_success_cnt) + " success")