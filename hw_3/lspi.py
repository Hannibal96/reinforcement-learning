import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer
from tqdm import tqdm

import matplotlib.pyplot as plt

def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    # compute the next w given the data.

    data_subset_idx = np.arange(len(actions))
    greedy_actions = linear_policy.get_max_action(encoded_next_states[data_subset_idx])
    phi_next_greedy = linear_policy.get_q_features(encoded_next_states[data_subset_idx], greedy_actions)
    phi_next_greedy[done_flags[data_subset_idx] == 1] = 0
    phi_current = linear_policy.get_q_features(encoded_states[data_subset_idx], actions[data_subset_idx])
    diff = (phi_current - gamma*phi_next_greedy).reshape((len(encoded_states[data_subset_idx]), 1, -1))
    c_hat = np.zeros((phi_current.shape[1], phi_current.shape[1]))
    for i in tqdm(range(len(encoded_states[data_subset_idx]))):
        curr = phi_current[i, :].reshape(-1, 1) @ diff[i, :]
        c_hat += curr

    d = np.sum(phi_current * rewards[data_subset_idx].reshape(-1, 1), axis=0)
    new_w = np.linalg.inv(c_hat) @ d
    return new_w.reshape(-1, 1)


if __name__ == '__main__':
    samples_to_collect = 100_000
    # samples_to_collect = 150000
    # samples_to_collect = 10000
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.999
    w_updates = 10
    evaluation_number_of_games = 50
    evaluation_max_steps_per_game = 1000

    seed_vs_success = False
    data_vs_success = True

    if seed_vs_success:
        seeds = [123, 234, 1]
        seed_sucess = {}
        for seed in seeds:
            # np.random.seed(123)
            # np.random.seed(234)
            print("*"*10, " Seed:", seed, "*"*10)
            np.random.seed(seed)
            seed_sucess[seed] = [0] * (w_updates+1)

            env = MountainCarWithResetEnv()
            # collect data
            states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
            # get data success rate
            data_success_rate = np.sum(rewards) / len(rewards)
            print(f'success rate {data_success_rate}')
            seed_sucess[seed][0] = data_success_rate
            # standardize data
            data_transformer = DataTransformer()
            data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
            states = data_transformer.transform_states(states)
            next_states = data_transformer.transform_states(next_states)
            # process with radial basis functions
            feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
            # encode all states:
            encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
            encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
            # set a new linear policy
            linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
            # but set the weights as random
            linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
            # start an object that evaluates the success rate over time
            evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
            for lspi_iteration in range(w_updates):
                print(f'starting lspi iteration {lspi_iteration}')

                new_w = compute_lspi_iteration(
                    encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
                )
                norm_diff = linear_policy.set_w(new_w)
                curr_rate_success = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
                seed_sucess[seed][lspi_iteration+1] = curr_rate_success
                if norm_diff < 0.00001:
                    for i in range(lspi_iteration+1, w_updates+1):
                        seed_sucess[seed][i] = curr_rate_success
                    break

            print(f'done lspi')
        #evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
        #evaluator.play_game(evaluation_max_steps_per_game, render=True)

        averaged_rate_success = np.zeros([w_updates+1])
        for k, v in seed_sucess.items():
            averaged_rate_success += np.array(v)
        averaged_rate_success = averaged_rate_success/len(seed_sucess)
        plt.plot(averaged_rate_success)
        plt.title("<rate success> vs iteration")
        plt.xlabel('Iteration')
        plt.ylabel('<rate success>')
        plt.show()


    evaluation_number_of_games = 100
    if data_vs_success:
        data_sucess = {}
        data_samples = range(10, 80, 5)
        for samples in data_samples:
            np.random.seed(123)
            samples_to_collect = 1000*samples
            data_sucess[samples_to_collect] = [0] * len(data_samples)

            print("*"*10,samples_to_collect, " Samples", "*"*10)

            env = MountainCarWithResetEnv()
            # collect data
            states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
            # get data success rate
            data_success_rate = np.sum(rewards) / len(rewards)
            print(f'success rate {data_success_rate}')
            # standardize data
            data_transformer = DataTransformer()
            data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
            states = data_transformer.transform_states(states)
            next_states = data_transformer.transform_states(next_states)
            # process with radial basis functions
            feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
            # encode all states:
            encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
            encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
            # set a new linear policy
            linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
            # but set the weights as random
            linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
            # start an object that evaluates the success rate over time
            evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
            for lspi_iteration in range(w_updates):
                print(f'starting lspi iteration {lspi_iteration}')

                new_w = compute_lspi_iteration(
                    encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
                )
                norm_diff = linear_policy.set_w(new_w)
                #curr_rate_success = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
                #seed_sucess[lspi_iteration] = curr_rate_success
                if norm_diff < 0.00001:
                    break

                # print('lspi iteration #:', lspi_iteration+1, ", rate success:", curr_rate_success)
            print(f'done lspi')
            final_rate_success = evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
            data_sucess[samples_to_collect] = final_rate_success
            #if final_rate_success == 1.0:
            #    evaluator.play_game(evaluation_max_steps_per_game, render=True)

        plt.plot(data_sucess.keys(), data_sucess.values())
        plt.title("<rate success> vs samples")
        plt.xlabel("Samples")
        plt.ylabel("<rate success>")
        plt.show()

