"""
Script pour visualiser les résultats après l'entraînement
Usage: python visualize_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comprehensive_results():
    """Génère des plots détaillés des résultats"""
    
    # Charger les rewards
    if not os.path.exists("logs/rewards.npy"):
        print("❌ Fichier logs/rewards.npy non trouvé!")
        print("   Lancez d'abord l'entraînement: python train.py")
        return
    
    rewards = np.load("logs/rewards.npy")
    print(f"✓ Chargement de {len(rewards)} épisodes")
    
    # Créer une figure avec plusieurs sous-plots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Courbe des rewards avec moyennes mobiles
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rewards, alpha=0.3, linewidth=0.5, label='Rewards bruts', color='lightblue')
    
    # Moyennes mobiles
    for window, color in [(10, 'orange'), (50, 'red'), (100, 'darkred')]:
        if len(rewards) >= window:
            moving_avg = [np.mean(rewards[max(0, i-window):i+1]) 
                          for i in range(len(rewards))]
            ax1.plot(moving_avg, linewidth=2, label=f'Moyenne mobile ({window})', color=color)
    
    ax1.set_xlabel('Épisode', fontsize=12)
    ax1.set_ylabel('Reward Total', fontsize=12)
    ax1.set_title('Progression des Rewards', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution des rewards
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Moyenne: {np.mean(rewards):.1f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--', 
                linewidth=2, label=f'Médiane: {np.median(rewards):.1f}')
    ax2.set_xlabel('Reward', fontsize=12)
    ax2.set_ylabel('Fréquence', fontsize=12)
    ax2.set_title('Distribution des Rewards', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Boxplot par tranches de 100 épisodes
    ax3 = plt.subplot(2, 2, 3)
    num_boxes = min(10, len(rewards) // 100)
    if num_boxes > 0:
        box_data = [rewards[i*100:(i+1)*100] for i in range(num_boxes)]
        bp = ax3.boxplot(box_data, labels=[f'{i*100}-{(i+1)*100}' for i in range(num_boxes)],
                         patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_xlabel('Épisodes', fontsize=12)
        ax3.set_ylabel('Reward', fontsize=12)
        ax3.set_title('Évolution par Tranches de 100 Épisodes', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Statistiques textuelles
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    stats_text = f"""
╔════════════════════════════════════════╗
║     📊 STATISTIQUES GLOBALES           ║
╚════════════════════════════════════════╝

Nombre d'épisodes: {len(rewards)}

Reward moyen:      {np.mean(rewards):>8.2f}
Reward médian:     {np.median(rewards):>8.2f}
Écart-type:        {np.std(rewards):>8.2f}

Reward minimum:    {np.min(rewards):>8.2f}
Reward maximum:    {np.max(rewards):>8.2f}

25e percentile:    {np.percentile(rewards, 25):>8.2f}
75e percentile:    {np.percentile(rewards, 75):>8.2f}

╔════════════════════════════════════════╗
║     📈 DERNIERS 100 ÉPISODES           ║
╚════════════════════════════════════════╝

Moyenne:           {np.mean(rewards[-100:]):>8.2f}
Minimum:           {np.min(rewards[-100:]):>8.2f}
Maximum:           {np.max(rewards[-100:]):>8.2f}

Amélioration:      {((np.mean(rewards[-100:]) - np.mean(rewards[:100])) / abs(np.mean(rewards[:100])) * 100 if len(rewards) >= 100 else 0):>7.1f}%
    """
    
    ax4.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = 'logs/comprehensive_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot sauvegardé: {output_file}")
    
    # Afficher
    plt.show()

def plot_simple():
    """Plot simple et rapide"""
    if not os.path.exists("logs/rewards.npy"):
        print("❌ Fichier logs/rewards.npy non trouvé!")
        return
    
    rewards = np.load("logs/rewards.npy")
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.5, label='Reward')
    
    # Moyenne mobile
    window = 50
    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) 
                  for i in range(len(rewards))]
    plt.plot(moving_avg, linewidth=2, label=f'Moyenne mobile ({window})', color='red')
    
    plt.xlabel('Épisode')
    plt.ylabel('Reward')
    plt.title('Courbe d\'Entraînement DQN - CARLA')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('logs/simple_plot.png', dpi=100)
    print("✓ Plot simple sauvegardé: logs/simple_plot.png")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("VISUALISATION DES RÉSULTATS")
    print("="*60)
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