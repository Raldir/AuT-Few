import os
import argparse
import numpy as np

import torch


def compute_kl_divergence(P, Q):
    return (P * (P / Q).log2()).sum()   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--experiment_name_1', type=str)
    parser.add_argument('--experiment_name_2', type=str)

    parser.add_argument('--majority_voting', help='override save', action='store_true')

    args = parser.parse_args()

    datasets = ["rte", "wic", "wsc", "anli-r1", "anli-r2", "anli-r3"]
    seeds = [42, 1024, 0, 1, 32]
    efficient_finetune_1 = "ia3"
    efficient_finetune_2 = "ia3"

    kl_divergence = []
    for dataset in datasets:
        kl_divergence_per_dataset = []
        for seed in seeds:
            num_shot = 32

            if "anli" in dataset:
                num_shot = 50

            save_path_1 = f'{args.experiment_name_1}_{dataset}_{seed}_{seed}_{num_shot}_{efficient_finetune_1}'
            save_path_1 = os.path.join("experiments", args.experiment_name_1, save_path_1, f"prediction_probs_dev_{dataset}_{args.majority_voting}_{seed}.csv")

            save_path_2 = f'{args.experiment_name_2}_{dataset}_{seed}_{seed}_{num_shot}_{efficient_finetune_2}'
            save_path_2 = os.path.join("experiments", args.experiment_name_2, save_path_2, f"prediction_probs_dev_{dataset}_{args.majority_voting}_{seed}.csv")

            P = np.genfromtxt(save_path_1, delimiter=",")
            Q = np.genfromtxt(save_path_2, delimiter=",")

            # P_swap = P[:, [1,0]]
            # Q = P_swap


            P = torch.tensor(P)
            Q = torch.tensor(Q)

            P_probs = torch.nn.functional.softmax(P, dim=1)
            Q_probs = torch.nn.functional.softmax(Q, dim=1)

            m = 0.5 * (Q_probs + P_probs)

            kl_divergence_1 = compute_kl_divergence(P_probs, m)
            kl_divergence_2 = compute_kl_divergence(Q_probs, m)

            jenson_shanon_divergence = (0.5 * kl_divergence_1) + (0.5 * kl_divergence_2)
            jenson_shanon_divergence = jenson_shanon_divergence.item() / P.size(0)
            kl_divergence_per_dataset.append(jenson_shanon_divergence)

        kl_divergence.append(sum(kl_divergence_per_dataset) / len(seeds))
    average_kl_divergence = sum(kl_divergence) / len(kl_divergence)
    print(average_kl_divergence)
    with open("temp.txt", "w") as f_out:
        f_out.write('\t'.join([str(x) for x in kl_divergence]))
    # print(' '.join([str(x) for x in kl_divergence]))






