from stable_baselines3.common.utils import get_device
import numpy as np
from middle_layer_network import MiddleLayerActorCriticPolicy

def normalize_obs(obs, obs_mean, obs_var):
    return (obs - obs_mean) / np.sqrt(obs_var + 1e-8)

if __name__ == '__main__':
    # setting
    device = "cuda"
    policy_path = "/home/haru/ros/jsk_aerial_robot_ws/src/jsk_aerial_robot/robots"+\
        "/agile_multirotor/gazebo_policy/RecurrentPPO_252"
    exec_max_gain = 3.0
    lstm_states = None

    print("policy_path: ", policy_path)
    policy_dir = policy_path  + "/policy.pth"

    act_mean = np.array([0.0, 0.0])[np.newaxis, :]
    act_std = np.array([exec_max_gain, exec_max_gain])[np.newaxis, :]

    rms_dir = policy_path + "/rms.npz"
    rms_data = np.load(rms_dir)
    obs_mean = np.mean(rms_data["mean"], axis=0)
    obs_var = np.mean(rms_data["var"], axis=0)

    # Create policy object
    policy = MiddleLayerActorCriticPolicy.load(policy_dir, device = device)

    # execute policy
    lstm_states = None
    obs = np.array([
        3.25000000e-01,  2.79338504e-01,  3.00000000e+00,  2.08543131e-01,
        -7.18291892e-02, -9.76831587e-04, -2.66088510e-04, -3.25075776e-03,
        2.87115809e-03, -5.26529137e-03, -5.36185545e-02,  8.19201314e-02,
        -1.77894185e-02, -3.54358500e-01, -3.53985042e-01,  1.74911451e+00,
        1.85022367e+00,  1.91393166e+00,  1.94478725e+00,  1.94478718e+00,
        1.91393143e+00,  1.85022327e+00,  1.74911391e+00,  1.60202576e+00,
        1.39273510e+00,  1.08731290e+00,  5.96972058e-01, -0.00000000e+00,
        -0.00000000e+00,  5.96601782e-01,  1.08694127e+00,  1.39236286e+00,
        1.60165316e+00,  1.74874105e+00,  1.84985021e+00,  1.91355820e+00,
        1.94441380e+00,  1.94441372e+00,  1.91355797e+00,  1.84984981e+00,
        1.74874046e+00, -5.52884063e-06, -3.13476958e-06, -1.77629670e-06,
        -4.42384794e-07,  4.42384794e-07,  1.77629670e-06,  3.13476958e-06,
        5.52886826e-06
    ])
    norm_obs = normalize_obs(obs, obs_mean, obs_var)
    n_act, lstm_states = policy.predict(norm_obs, state = lstm_states, deterministic=True)
    print("action ros obs: ", n_act)

    lstm_states = None
    obs = np.array([
        0.325, 0.279339, 3, 1,
        -0.0571276, 7.86363e-16, 2.16464e-14, -2.41752e-15,
        9.48276e-15, -0.0668776, 0.0622051, -0.249927,
        0.0496948, -0.354172, -0.354172, 1.74893,
        1.85004, 1.91375, 1.9446, 1.9446,
        1.91375, 1.85004, 1.74893, 1.60184,
        1.39255, 1.08713, 0.596787, -0,
        -0, 0.596787, 1.08713, 1.39255,
        1.60184, 1.74893, 1.85004, 1.91375,
        1.9446, 1.9446, 1.91375, 1.85004,
        1.74893, -2.4603e-20, -1.39495e-20, -7.90439e-21,
        -1.96858e-21, 1.96858e-21, 7.90439e-21, 1.39495e-20,
        2.4603e-20
    ])
    norm_obs = normalize_obs(obs, obs_mean, obs_var)
    n_act, lstm_states = policy.predict(norm_obs, state = lstm_states, deterministic=True)
    print("action flightmare obs: ", n_act)