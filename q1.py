import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Neural Population Dynamics
# ------------------------------
np.random.seed(42)
N = 1000
# Trial blocks
K = 5
# Events
M = 10
sigma_pf = 0.1

pref_positions = np.random.uniform(0, 1, size=N)

def place_field(x, pref, sigma):
    return np.exp(-((x - pref)**2) / (2 * sigma**2))

trial_positions = np.linspace(0.2, 0.8, K)
r_templates = np.array([np.array([place_field(x, p, sigma_pf) for p in pref_positions]) for x in trial_positions])

# -------------------------------------
# 2. Low-Dimensional Manifold Embedding
# -------------------------------------
manifold_dim = 2
z_templates = np.array([[np.cos(2 * np.pi * k / K), np.sin(2 * np.pi * k / K)] for k in range(K)])

# -----------------------------------
# 3. SPW-R Generation via Diffusion
# -----------------------------------
def simulate_diffusion(T_steps, dt, v, diff_coeff, z0):
    traj = [z0]
    for _ in range(T_steps):
        noise = np.random.normal(scale=np.sqrt(diff_coeff * dt), size=z0.shape)
        z_new = traj[-1] + v * dt + noise
        traj.append(z_new)
    return np.array(traj)

T_steps = 50
dt = 0.1
v = np.array([0.05, 0.05])
diff_coeff = 0.1
beta = 5.0
sigma_spwr = 0.05

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# Simulation phase
spwr_events = []
selected_trial_blocks = []
manifold_final_states = []
w = np.zeros((N, N))
eta = 0.01

for m in range(M):
    z0 = np.random.uniform(-1, 1, size=(manifold_dim,))
    traj = simulate_diffusion(T_steps, dt, v, diff_coeff, z0)
    z_final = traj[-1]
    manifold_final_states.append(z_final)
    
    sims = np.array([cosine_similarity(z_final, z_temp) for z_temp in z_templates])
    p_k = np.exp(beta * sims)
    p_k /= np.sum(p_k)
    selected_k = np.random.choice(np.arange(K), p=p_k)
    selected_trial_blocks.append(selected_k)
    
    s_m = r_templates[selected_k] + np.random.normal(scale=sigma_spwr, size=r_templates[selected_k].shape)
    spwr_events.append(s_m)
    
    w += eta * p_k[selected_k] * np.outer(s_m, s_m)

# Decoding
decoded_trial_blocks = []
for z in manifold_final_states:
    distances = np.linalg.norm(z_templates - z, axis=1)
    decoded_k = np.argmin(distances)
    decoded_trial_blocks.append(decoded_k)

# Wake-sleep distributions
wake_counts = np.zeros(K)
for k in selected_trial_blocks:
    wake_counts[k] += 1
p_wake = wake_counts / np.sum(wake_counts)
gamma = 2.0
p_sleep = p_wake**gamma
p_sleep /= np.sum(p_sleep)

# -------------------------------
# Visualization with Subplots
# -------------------------------
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# Manifold Trajectories
ax1 = fig.add_subplot(gs[0, 0])
for m in range(M):
    z0 = np.random.uniform(-1, 1, size=(manifold_dim,))
    traj = simulate_diffusion(T_steps, dt, v, diff_coeff, z0)
    ax1.plot(traj[:, 0], traj[:, 1], alpha=0.8, label=f'Event {m+1}' if m == 0 else "")
ax1.scatter(z_templates[:, 0], z_templates[:, 1], color='red', marker='x', s=100, label='Templates')
ax1.set_xlabel('Manifold Dim 1', fontsize=12, weight = 'bold')
ax1.set_ylabel('Manifold Dim 2', fontsize=12, weight = 'bold')
ax1.set_title('(A) Manifold Dynamics', fontsize=14, weight = 'bold')
ax1.legend()

# Spike Patterns Heatmap
ax2 = fig.add_subplot(gs[0, 1])
#num_events_plot = 3
#num_neurons_plot = 50
#spike_data = np.array(spwr_events[:num_events_plot])[:, :num_neurons_plot].T
im2 = ax2.imshow(spwr_events, aspect='auto', cmap='viridis')
ax2.set_xlabel('SpWR Event', fontsize=12, weight = 'bold')
ax2.set_ylabel('Neuron Index', fontsize=12, weight = 'bold')
ax2.set_title('(B) Spike Patterns Events', fontsize=14, weight = 'bold')
fig.colorbar(im2, ax=ax2)

# Cosine Similarity Heatmap
ax3 = fig.add_subplot(gs[0, 2])
sim_matrix = np.array([[cosine_similarity(z, z_temp) for z_temp in z_templates] for z in manifold_final_states])
im3 = ax3.imshow(sim_matrix, aspect='auto', cmap='viridis')
ax3.set_xlabel('Trial Block', fontsize=12, weight = 'bold')
ax3.set_ylabel('SpWR Event', fontsize=12, weight = 'bold')
ax3.set_title('(C) Cosine Similarity', fontsize=14, weight = 'bold')
fig.colorbar(im3, ax=ax3)
ax3.set_xticks(np.arange(K))
ax3.set_yticks(np.arange(M))
ax3.set_xticklabels([f'TB{k+1}' for k in range(K)])
ax3.set_yticklabels([f'E{m+1}' for m in range(M)])

# Decoding Comparison
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(np.arange(M), selected_trial_blocks, marker='o', label='Selected', c='blue')
ax4.scatter(np.arange(M), decoded_trial_blocks, marker='x', label='Decoded', c='red')
ax4.set_xlabel('SpWR Event', fontsize=12, weight = 'bold')
ax4.set_ylabel('Trial Block', fontsize=12, weight = 'bold')
ax4.set_title('(D) Selected vs Decoded Trial Blocks', fontsize=14, weight = 'bold')
ax4.set_yticks([0, 1, 2])
ax4.legend()

# Wake vs Sleep Distributions
ax5 = fig.add_subplot(gs[1, 1])
x = np.arange(K)
width = 0.4
ax5.bar(x - width/2, p_wake, width, label='Wake', alpha=0.7)
ax5.bar(x + width/2, p_sleep, width, label='Sleep', alpha=0.7)
ax5.set_xlabel('Trial Block',fontsize=12, weight = 'bold')
ax5.set_ylabel('Probability',fontsize=12, weight = 'bold')
ax5.set_title('(E) Trial Block Distributions', fontsize=14, weight = 'bold')
ax5.set_xticks(x)
ax5.legend()

# Synaptic Weights Histogram
ax6 = fig.add_subplot(gs[1, 2])
weights = w.flatten()
ax6.hist(weights, bins=50, log=True, color='gray', edgecolor='black')
ax6.set_xlabel('Synaptic Weight', fontsize=12, weight = 'bold')
ax6.set_ylabel('Count (log scale)', fontsize=12, weight = 'bold')
ax6.set_title('(F) Synaptic Weight Distribution', fontsize=14, weight = 'bold')

plt.tight_layout()
plt.savefig('result_1.jpg', dpi = 600)
plt.show()