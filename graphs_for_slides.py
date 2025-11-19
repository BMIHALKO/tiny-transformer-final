import json, math, os
import matplotlib.pyplot as plt
import numpy as np

# --- Load loss log ---
with open("./tiny_ckpt/FinalCheckpoint/final_metrics.json", "r", encoding="utf-8") as f:
    logs = json.load(f)

# --- Extract data ---
train_steps  = logs.get("train_steps", [])
train_losses = logs.get("train_losses", [])
valid_steps  = logs.get("valid_steps", [])
valid_losses = logs.get("valid_losses", [])
bleu_steps   = logs.get("bleu_steps", [])
bleu_scores  = logs.get("bleu_scores", [])
final_bleu   = logs.get("final_bleu", None)
best_valid   = logs.get("best_valid_loss", None)

# --- Compute PPL if needed ---
valid_ppl = [math.exp(l) for l in valid_losses]

# --- Create save directory ---
save_dir = "finalCheckpoint_graphs"
os.makedirs(save_dir, exist_ok=True)

# --------------------------------------------------------
# 1. Training + Validation Loss Plot (Annotated)
# --------------------------------------------------------
plt.figure(figsize=(9,5))

plt.plot(train_steps, train_losses, label='Train Loss', color='#e67e22')
plt.plot(valid_steps, valid_losses, label='Validation Loss', color='#2980b9')


# Highlight best validation loss
if best_valid is not None:
    best_idx = np.argmin(valid_losses)
    best_step = valid_steps[best_idx]
    plt.scatter(best_step, best_valid, color='red', zorder=10)
    plt.annotate(f'Best val loss = {best_valid:.3f}\n(step {best_step})',
                 xy=(best_step, best_valid),
                 xytext=(best_step + 500, best_valid + 0.5),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9)


plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=200)
plt.close()

# --------------------------------------------------------
# 2. BLEU Curve (Annotated)
# --------------------------------------------------------
plt.figure(figsize=(9,5))

plt.plot(bleu_steps, bleu_scores, marker='o', color='#27ae60', label='BLEU Score')

# Mark final BLEU
if final_bleu is not None:
    plt.scatter(bleu_steps[-1], final_bleu, color='red', zorder=10)
    plt.annotate(f'Final BLEU = {final_bleu:.2f}',
                 xy=(bleu_steps[-1], final_bleu),
                 xytext=(bleu_steps[-1] - 1500, final_bleu + 0.5),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9)


plt.xlabel('Steps')
plt.ylabel('BLEU')
plt.title('BLEU Score Progression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'bleu_curve.png'), dpi=200)
plt.close()


# --------------------------------------------------------
# 3. Validation Perplexity
# --------------------------------------------------------
plt.figure(figsize=(9,5))
plt.plot(valid_steps, valid_ppl, marker='s', color='#8e44ad')
plt.xlabel('Steps')
plt.ylabel('Perplexity')
plt.title('Validation Perplexity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'ppl_curve.png'), dpi=200)
plt.close()

# --------------------------------------------------------
# 4. (Optional) Combined 3-Panel Figure for the Paper
# --------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(9, 12))

# Panel 1: Loss
axs[0].plot(train_steps, train_losses, color='#e67e22', label='Train Loss')
axs[0].plot(valid_steps, valid_losses, color='#2980b9', label='Validation Loss')
axs[0].set_title("Training + Validation Loss")
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Panel 2: BLEU
axs[1].plot(bleu_steps, bleu_scores, color='#27ae60', marker='o')
axs[1].set_title("BLEU Score Progression")
axs[1].grid(True, alpha=0.3)

# Panel 3: Perplexity
axs[2].plot(valid_steps, valid_ppl, color='#8e44ad', marker='s')
axs[2].set_title("Validation Perplexity")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'all_metrics_combined.png'), dpi=200)
plt.close()


print("✅ All updated plots saved!")
print("   • loss_curve.png")
print("   • bleu_curve.png")
print("   • ppl_curve.png")
print("   • all_metrics_combined.png")