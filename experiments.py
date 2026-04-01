import os
import numpy as np
import matplotlib.pyplot as plt

from dqn_cartpole import Config, train


def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def run_multiple_seeds(cfg, seeds):
    all_rewards = []
    for seed in seeds:
        cfg.seed = seed
        rewards = train(cfg)
        all_rewards.append(rewards)
    return all_rewards


def align_curves(curves):
    min_len = min(len(c) for c in curves)
    trimmed = np.array([c[:min_len] for c in curves], dtype=float)
    return trimmed


def plot_comparison(results_dict, save_path="results/compare_configs.png"):
    plt.figure(figsize=(10, 6))

    for label, curves in results_dict.items():
        curves = align_curves(curves)

        smoothed = []
        for c in curves:
            smoothed_curve = moving_average(c, window=10)
            smoothed.append(smoothed_curve)

        smoothed = align_curves(smoothed)
        mean_curve = smoothed.mean(axis=0)
        std_curve = smoothed.std(axis=0)

        x = np.arange(len(mean_curve))
        plt.plot(x, mean_curve, label=label)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Comparison of DQN Configurations on CartPole")
    plt.legend()
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]

    results = {}

    print("Running Naive...")
    results["Naive"] = run_multiple_seeds(
        Config(use_target=False, use_replay=False),
        seeds
    )

    print("Running Only TN...")
    results["Only TN"] = run_multiple_seeds(
        Config(use_target=True, use_replay=False),
        seeds
    )

    print("Running Only ER...")
    results["Only ER"] = run_multiple_seeds(
        Config(use_target=False, use_replay=True),
        seeds
    )

    print("Running TN + ER...")
    results["TN + ER"] = run_multiple_seeds(
        Config(use_target=True, use_replay=True),
        seeds
    )

    plot_comparison(results)
    print("Done. Saved to results/compare_configs.png")