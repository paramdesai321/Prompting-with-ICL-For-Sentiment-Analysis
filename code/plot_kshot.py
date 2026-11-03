import matplotlib.pyplot as plt

eval_results = {
    0: 77.98,
    1: 83.83,
    3: 84.17,
    5: 86.24,
    8: 89.33,
    16: 88.65
}

k_values = list(eval_results.keys())
accuracies = list(eval_results.values())

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linewidth=2.5, markersize=7, color='#1f77b4', label='Qwen3-0.6B-Base')

plt.title('SST-2 Validation Accuracy vs. In-Context Exemplars (K)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of In-Context Exemplars (K)', fontsize=11, labelpad=8)
plt.ylabel('Validation Accuracy (%)', fontsize=11, labelpad=8)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)

plt.ylim(70, 95)

for k, acc in eval_results.items():
    plt.annotate(f"{acc:.2f}%", (k, acc), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
