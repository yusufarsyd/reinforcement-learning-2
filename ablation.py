import os
import numpy as np
import matplotlib.pyplot as plt

from dqn_cartpole import Config, train


def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def align_curves(curves):
    min_len = min(len(c) for c in curves)
    return np.array([c[:min_len] for c in curves], dtype=float)


def run_multiple_seeds(cfg, seeds):
    all_rewards = []
    for seed in seeds:
        new_cfg = Config(**vars(cfg))
        new_cfg.seed = seed
        rewards = train(new_cfg)
        all_rewards.append(rewards)
    return all_rewards


def plot_ablation(results_dict, title, save_path):
    plt.figure(figsize=(10, 6))

    for label, curves in results_dict.items():
        curves = align_curves(curves)

        smoothed = []
        for c in curves:
            smoothed.append(moving_average(c, window=10))

        smoothed = align_curves(smoothed)
        mean_curve = smoothed.mean(axis=0)
        std_curve = smoothed.std(axis=0)

        x = np.arange(len(mean_curve))
        plt.plot(x, mean_curve, label=label)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def run_learning_rate_ablation(seeds, total_steps=20000):
    results = {
        "lr=1e-4": run_multiple_seeds(
            Config(lr=1e-4, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "lr=5e-4": run_multiple_seeds(
            Config(lr=5e-4, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "lr=1e-3": run_multiple_seeds(
            Config(lr=1e-3, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
    }
    plot_ablation(results, "Learning Rate Ablation", "results/ablation_learning_rate.png")


def run_exploration_ablation(seeds, total_steps=20000):
    results = {
        "eps_end=0.01": run_multiple_seeds(
            Config(epsilon_end=0.01, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "eps_end=0.05": run_multiple_seeds(
            Config(epsilon_end=0.05, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "eps_end=0.10": run_multiple_seeds(
            Config(epsilon_end=0.10, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
    }
    plot_ablation(results, "Exploration Ablation", "results/ablation_exploration.png")


def run_network_size_ablation(seeds, total_steps=20000):
    results = {
        "(32,32)": run_multiple_seeds(
            Config(hidden_sizes=(32, 32), total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "(64,64)": run_multiple_seeds(
            Config(hidden_sizes=(64, 64), total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "(128,128)": run_multiple_seeds(
            Config(hidden_sizes=(128, 128), total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
    }
    plot_ablation(results, "Network Size Ablation", "results/ablation_network_size.png")


def run_update_ratio_ablation(seeds, total_steps=20000):
    results = {
        "ups=1": run_multiple_seeds(
            Config(updates_per_step=1, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "ups=2": run_multiple_seeds(
            Config(updates_per_step=2, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
        "ups=4": run_multiple_seeds(
            Config(updates_per_step=4, total_steps=total_steps, use_target=True, use_replay=True),
            seeds
        ),
    }
    plot_ablation(results, "Update-to-Data Ratio Ablation", "results/ablation_update_ratio.png")


if __name__ == "__main__":
    seeds = [1, 2]

    print("Running learning rate ablation...")
    run_learning_rate_ablation(seeds, total_steps=20000)

    print("Running exploration ablation...")
    run_exploration_ablation(seeds, total_steps=20000)

    print("Running network size ablation...")
    run_network_size_ablation(seeds, total_steps=20000)

    print("Running update ratio ablation...")
    run_update_ratio_ablation(seeds, total_steps=20000)

    print("Done. All ablation plots saved in results/")