import time
import argparse
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch import tensor

from btor2select_env import Btor2SelectMDP_Discrete
from parse_raw_tsv import parse_from_tsv_lst
from analyze import get_par_N, delete_unsolvables_from_dict
from create_feature import KEYWORDS, read_btor2kw_csv

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# now the scaler is based on all instances; change later
def add_scaled_kw(kw_embds):
    all_embeds = [value_tuple[1] for value_tuple in kw_embds.values()]
    embeds_matrix = np.array(all_embeds)
    scaler = MinMaxScaler()
    scaled_embeds_matrix = scaler.fit_transform(embeds_matrix)
    scaled_kw_embds = {}
    for i, (key, value_tuple) in enumerate(kw_embds.items()):
        scaled_kw_embds[key] = (value_tuple[0], value_tuple[1], scaled_embeds_matrix[i])
    return scaled_kw_embds, scaler

def main():
    parser = argparse.ArgumentParser(description='Using dqn to play Btor2Select')
    parser.add_argument("-f", "--fixed", action="store_true", help="Use a fixed policy (default is False)")
    parser.add_argument("-s", "--seed", type=int, help="Random seed")
    parser.add_argument("-m", "--model-save-path", type=str, help="Path to save the trained model", required=True)
    parser.add_argument('input_files', nargs='+', help="List of input performance result files")
    
    args = parser.parse_args()
    is_fixed = args.fixed
    seed = args.seed
    np.random.seed(seed)
    model_save_path = args.model_save_path
    training_res_tsv_lst = args.input_files

    # training_res_tsv_lst = ["bv_data/aaai25/5fold/4.tsv", "bv_data/aaai25/5fold/0.tsv", "bv_data/aaai25/5fold/1.tsv", "bv_data/aaai25/5fold/2.tsv"]
    training_res_dict, tool_config_dict = parse_from_tsv_lst(training_res_tsv_lst, num_tool_cols=2, timeout=900.0)

    solvable_training_res_dict = delete_unsolvables_from_dict(training_res_dict)
    # tool_config_size = len(tool_config_dict)

    kw_size = len(KEYWORDS)
    kw_embds = read_btor2kw_csv("benchmarks/bv/bv_btor2kw.csv")
    sc_kw_embds, kw_scaler = add_scaled_kw(kw_embds)

    env = Btor2SelectMDP_Discrete(
                        instance_embed_size=kw_size,
                        res_dict=solvable_training_res_dict,
                        tool_config_dict=tool_config_dict,
                        early_timeouts=[100, 300],
                        embd_dict=sc_kw_embds,
                        timeout=900
                    )
    check_env(env)
    env = Monitor(env, filename="dqn.log")

    eval_callback = EvalCallback(
                        env,
                        best_model_save_path=model_save_path, 
                        log_path="dqn_logs/", 
                        eval_freq=1000,
                        n_eval_episodes=1000,
                        deterministic=True,
                        render=False
                    )

    # Check the environment
    model = DQN("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=500000, log_interval=4, callback=eval_callback)
    # model.save(model_save_path)

if __name__ == "__main__":
    main()