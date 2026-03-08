"""
Script pour visualiser les résultats après l'entraînement
Usage: python visualize_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window=50):
    return [np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))]

def load_log(path):
    """Load a .npy log file if it exists, else return None."""
    if os.path.exists(path):
        return np.load(path)
    return None

def plot_comprehensive_results():
    """Génère des plots détaillés des résultats"""

    if not os.path.exists("logs/rewards.npy"):
        print("❌ Fichier logs/rewards.npy non trouvé!")
        print("   Lancez d'abord l'entraînement: python train.py")
        return

    # Load all logs
    rewards        = load_log("logs/rewards.npy")
    collisions     = load_log("logs/collision_rate.npy")
    ep_lengths     = load_log("logs/episode_length.npy")
    avg_speeds     = load_log("logs/avg_speed.npy")
    avg_lane_devs  = load_log("logs/avg_lane_dev.npy")
    avg_losses     = load_log("logs/avg_loss.npy")
    mean_max_qs    = load_log("logs/mean_max_q.npy")
    epsilons       = load_log("logs/epsilon.npy")

    print(f"✓ Chargement de {len(rewards)} épisodes")

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Double DQN — CARLA Training Results", fontsize=16, fontweight='bold', y=0.98)

    # -------------------------------------------------------
    # 1. Reward progression
    # -------------------------------------------------------
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(rewards, alpha=0.3, linewidth=0.5, color='lightblue', label='Raw')
    for window, color in [(10, 'orange'), (50, 'red'), (100, 'darkred')]:
        if len(rewards) >= window:
            ax1.plot(moving_average(rewards, window), linewidth=2,
                     label=f'MA({window})', color=color)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Reward Progression', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------
    # 2. Reward distribution
    # -------------------------------------------------------
    ax2 = plt.subplot(4, 2, 2)
    ax2.hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--',
                linewidth=2, label=f'Median: {np.median(rewards):.1f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------
    # 3. Episode length (survival time)
    # -------------------------------------------------------
    ax3 = plt.subplot(4, 2, 3)
    if ep_lengths is not None:
        ax3.plot(ep_lengths, alpha=0.3, linewidth=0.5, color='lightgreen')
        ax3.plot(moving_average(ep_lengths, 50), linewidth=2,
                 color='green', label='MA(50)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps Survived')
        ax3.set_title('Episode Length (Survival Time)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax3.set_title('Episode Length', fontweight='bold')

    # -------------------------------------------------------
    # 4. Collision rate
    # -------------------------------------------------------
    ax4 = plt.subplot(4, 2, 4)
    if collisions is not None:
        ax4.plot(collisions, alpha=0.3, linewidth=0.5, color='salmon')
        ax4.plot(moving_average(collisions, 50), linewidth=2,
                 color='red', label='MA(50)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Collisions')
        ax4.set_title('Collision Rate', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax4.set_title('Collision Rate', fontweight='bold')

    # -------------------------------------------------------
    # 5. Average speed
    # -------------------------------------------------------
    ax5 = plt.subplot(4, 2, 5)
    if avg_speeds is not None:
        ax5.plot(avg_speeds, alpha=0.3, linewidth=0.5, color='lightyellow')
        ax5.plot(moving_average(avg_speeds, 50), linewidth=2,
                 color='goldenrod', label='MA(50)')
        ax5.axhline(20.0, color='blue', linestyle='--',
                    linewidth=1.5, label='Target speed (20 km/h)')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Speed (km/h)')
        ax5.set_title('Average Speed per Episode', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax5.set_title('Average Speed', fontweight='bold')

    # -------------------------------------------------------
    # 6. Lane deviation
    # -------------------------------------------------------
    ax6 = plt.subplot(4, 2, 6)
    if avg_lane_devs is not None:
        ax6.plot(avg_lane_devs, alpha=0.3, linewidth=0.5, color='plum')
        ax6.plot(moving_average(avg_lane_devs, 50), linewidth=2,
                 color='purple', label='MA(50)')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Distance to Lane Center (m)')
        ax6.set_title('Average Lane Deviation', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax6.set_title('Lane Deviation', fontweight='bold')

    # -------------------------------------------------------
    # 7. Training loss
    # -------------------------------------------------------
    ax7 = plt.subplot(4, 2, 7)
    if avg_losses is not None:
        # Skip first few episodes where loss can be very noisy
        loss_data = avg_losses[10:]
        ax7.plot(range(10, len(avg_losses)), loss_data,
                 alpha=0.3, linewidth=0.5, color='lightcoral')
        ax7.plot(range(10, len(avg_losses)),
                 moving_average(loss_data, 50), linewidth=2,
                 color='darkred', label='MA(50)')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('MSE Loss')
        ax7.set_title('Training Loss', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax7.set_title('Training Loss', fontweight='bold')

    # -------------------------------------------------------
    # 8. Mean Max Q + Epsilon overlay
    # -------------------------------------------------------
    ax8 = plt.subplot(4, 2, 8)
    if mean_max_qs is not None:
        ax8.plot(moving_average(mean_max_qs, 50), linewidth=2,
                 color='steelblue', label='Mean Max Q — MA(50)')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Q-Value', color='steelblue')
        ax8.tick_params(axis='y', labelcolor='steelblue')
        ax8.set_title('Mean Max Q-Value & Epsilon', fontweight='bold')

        if epsilons is not None:
            ax8b = ax8.twinx()
            ax8b.plot(epsilons, linewidth=1.5, color='orange',
                      linestyle='--', label='Epsilon')
            ax8b.set_ylabel('Epsilon', color='orange')
            ax8b.tick_params(axis='y', labelcolor='orange')
            ax8b.set_ylim(0, 1.1)
            lines1, labels1 = ax8.get_legend_handles_labels()
            lines2, labels2 = ax8b.get_legend_handles_labels()
            ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax8.set_title('Mean Max Q-Value & Epsilon', fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = 'logs/comprehensive_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot sauvegardé: {output_file}")
    plt.show()

def plot_simple():
    """Plot simple et rapide"""
    if not os.path.exists("logs/rewards.npy"):
        print("❌ Fichier logs/rewards.npy non trouvé!")
        return

    rewards = np.load("logs/rewards.npy")

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.5, label='Reward')
    window = 50
    moving_avg = [np.mean(rewards[max(0, i - window):i + 1])
                  for i in range(len(rewards))]
    plt.plot(moving_avg, linewidth=2, label=f'MA({window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Double DQN Training Curve — CARLA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('logs/simple_plot.png', dpi=100)
    print("✓ Plot simple sauvegardé: logs/simple_plot.png")
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("VISUALISATION DES RÉSULTATS — Double DQN")
    print("=" * 60)
    print("\nChoisissez:")
    print("1. Plot complet (recommandé)")
    print("2. Plot simple")

    choice = input("\nVotre choix (1 ou 2): ").strip()

    if choice == "1":
        plot_comprehensive_results()
    elif choice == "2":
        plot_simple()
    else:
        print("Génération du plot complet par défaut...")
        plot_comprehensive_results()