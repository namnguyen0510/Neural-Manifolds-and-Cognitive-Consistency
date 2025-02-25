import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


# ============================
# 1. Data Preparation
# ============================
#np.random.seed(42)
#torch.manual_seed(42)

# Fixed neuron embeddings (reference points)
m_neurons = 50
neuron_embeddings = np.random.rand(m_neurons, 2)
kmeans = KMeans(n_clusters=3, random_state=42).fit(neuron_embeddings)
neuron_groups = kmeans.labels_

# Generate knowledge embeddings (training and test data)
n_train = 30
n_test = 20
X_train = np.random.normal(loc=0.5, scale=0.15, size=(n_train, 2))
X_test = np.random.uniform(low=0.0, high=1.0, size=(n_test, 2))
knowledge_embeddings = np.vstack([X_train, X_test])  # Combined for visualization

# Mapping function with distance calculation
def map_to_neurons(embeddings):
    mapping = []
    distances = []
    for point in embeddings:
        dists = np.linalg.norm(neuron_embeddings - point, axis=1)
        idx = np.argmin(dists)
        mapping.append(idx)
        distances.append(dists[idx])
    return np.array(mapping), np.array(distances)

# Create labels based on neuron clusters
mapping, mapping_distances = map_to_neurons(knowledge_embeddings)
y_train = neuron_groups[mapping[:n_train]]
y_test = neuron_groups[mapping[n_train:]]

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ============================
# 2. Synaptic Weight Optimization (Heider's Balance Theory)
# ============================
def compute_balance_energy(W):
    energy = 0.0
    m = W.shape[0]
    for i in range(m-2):
        for j in range(i+1, m-1):
            for k in range(j+1, m):
                energy += (W[i, j] * W[j, k] * W[k, i] + 1)**2
    return energy

# Initialize and optimize synaptic weights
A = np.random.randn(m_neurons, m_neurons)
W_init = (A + A.T) / 2.0
np.fill_diagonal(W_init, 0)
W = W_init.copy()
energy_history = []
num_iterations = 200
eta = 0.001

for _ in range(num_iterations):
    E = compute_balance_energy(W)
    energy_history.append(E)
    grad = np.zeros_like(W)
    m = W.shape[0]
    for i in range(m-2):
        for j in range(i+1, m-1):
            for k in range(j+1, m):
                term = W[i, j] * W[j, k] * W[k, i] + 1
                grad[i, j] += 2 * term * (W[j, k] * W[k, i])
                grad[j, i] = grad[i, j]
                grad[j, k] += 2 * term * (W[i, j] * W[k, i])
                grad[k, j] = grad[j, k]
                grad[k, i] += 2 * term * (W[i, j] * W[j, k])
                grad[i, k] = grad[k, i]
    W -= eta * grad
W_final = W.copy()

# ============================
# 3. Model Definitions and Training
# ============================
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        return self.fc(out.squeeze(1))

class HeiderModel(RNNModel):
    def __init__(self, input_dim, hidden_dim, output_dim, lambda_reg=0.01):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.lambda_reg = lambda_reg

    def compute_balance_energy(self):
        W = self.fc.weight
        energy = 0.0
        m = W.shape[0]
        for i in range(m-2):
            for j in range(i+1, m-1):
                for k in range(j+1, m):
                    term = W[i, j] * W[j, k] * W[k, i]
                    energy += (term + 1) ** 2
        return energy

# Training function with metrics
def train_model(model, criterion, optimizer, num_epochs=100):
    train_losses = []
    test_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        if isinstance(model, HeiderModel):
            loss += model.lambda_reg * model.compute_balance_energy()
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, pred = torch.max(test_outputs, 1)
            test_acc = (pred == y_test_t).sum().item() / len(y_test_t)
            test_accs.append(test_acc)
    
    return train_losses, test_accs

# ============================
# 6. Decision Boundary Visualization
# ============================

def plot_decision_boundary(model, X_train, y_train, title, ax):
    # Define grid boundaries based on data range
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create grid points as inputs
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    # Predict class labels for each grid point
    model.eval()
    with torch.no_grad():
        Z = model(grid_points)
        _, preds = torch.max(Z, 1)
        Z = preds.numpy().reshape(xx.shape)
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', edgecolor='k', s=100)
    ax.set_title(title,fontsize = 14, weight = "bold")
    ax.set_xlabel("Dimension 1",fontsize = 12, weight = "bold")
    ax.set_ylabel("Dimension 2",fontsize = 12, weight = "bold")

# Define the custom colors
colors = ['#e41a1c', '#377eb8', '#4daf4a']
# Create the colormap
custom_cmap = ListedColormap(colors, name="custom_cmap")
# Add decision boundary plots to the visualization function
def create_visualizations_with_decision_boundaries():
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    
    # --- Plot 1: Knowledge-to-Neuron Mapping ---
    ax = axs[0, 0]
    scatter = ax.scatter(neuron_embeddings[:, 0], neuron_embeddings[:, 1], 
                        c=neuron_groups, cmap=custom_cmap, marker='o', s=100, label='Neurons')
    ax.scatter(knowledge_embeddings[:, 0], knowledge_embeddings[:, 1], 
              c='gray', marker='o', s=50, label='Knowledge')
    for i, k in enumerate(knowledge_embeddings):
        neuron_idx = mapping[i]
        n_emb = neuron_embeddings[neuron_idx]
        ax.plot([k[0], n_emb[0]], [k[1], n_emb[1]], alpha=0.5)
    ax.set_title("(A) Knowledge-to-Neuron\nMapping", fontsize = 14, weight = "bold")
    ax.set_xlabel("Dimension 1", fontsize = 12, weight = "bold")
    ax.set_ylabel("Dimension 2", fontsize = 12, weight = "bold")
    legend_elements = [Patch(facecolor=c, label=f'Neuron Group {i}') 
                      for i, c in enumerate(['#e41a1c','#377eb8','#4daf4a'])]
    ax.legend(handles=legend_elements + [Patch(facecolor='gray', label='Knowledge')], loc='best')

    # --- Plot 2: Balance Energy Evolution ---
    ax = axs[0, 2]
    ax.plot(energy_history, 'b-', linewidth=2)
    ax.set_title("(B) Balance Energy Optimization",fontsize = 14, weight = "bold")
    ax.set_xlabel("Iteration",fontsize = 12, weight = "bold")
    ax.set_ylabel("Balance Energy",fontsize = 12, weight = "bold")
    ax.grid(True)

    # --- Plot 5: Network Graph Visualization ---
    ax = axs[0, 1]
    G = nx.Graph()
    for i in range(m_neurons):
        G.add_node(i, pos=neuron_embeddings[i], group=neuron_groups[i])
    threshold = np.percentile(np.abs(W_final[np.triu_indices(m_neurons, k=1)]), 95)
    for i in range(m_neurons-1):
        for j in range(i+1, m_neurons):
            if np.abs(W_final[i, j]) >= threshold:
                G.add_edge(i, j, weight=W_final[i, j])
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[i]['group'] for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, cmap=custom_cmap, node_size=150)
    edges = G.edges(data=True)
    edge_widths = [np.abs(data['weight']) for _, _, data in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color='gray')
    ax.set_title(f"(C) Final Neuron Network\n(Edges > {threshold:.2f})", fontsize = 14, weight = "bold")
    ax.set_axis_off()

    # --- Plot 6: Weight Distribution Comparison ---
    ax = axs[0, 3]
    init_weights = W_init[np.triu_indices(m_neurons, k=1)]
    final_weights = W_final[np.triu_indices(m_neurons, k=1)]
    ax.hist(init_weights/init_weights.sum(), bins=30, alpha=0.6, label='Initial', color = 'blue')
    ax.hist(final_weights/final_weights.sum(), bins=30, alpha=0.6, label='Optimized', color = 'red')
    ax.set_title("(D) Synaptic Weight\n Distribution Evolution",fontsize = 14, weight = "bold")
    ax.set_xlabel("Weight Value",fontsize = 12, weight = "bold")
    ax.set_ylabel("Probability",fontsize = 12, weight = "bold")
    ax.legend()

    # --- Plot 9: Neuron Embeddings ---
    ax = axs[1, 0]
    ax.scatter(neuron_embeddings[:, 0], neuron_embeddings[:, 1], 
              c=neuron_groups, cmap=custom_cmap, marker='o', s=100)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, label='Centers')
    ax.set_title("(E) Neural Group's Centroids",fontsize = 14, weight = "bold")
    ax.legend()

    # --- Plot 10: Neuron Group Counts ---
    ax = axs[1, 1]
    unique, counts = np.unique(neuron_groups, return_counts=True)
    col_names = [f"Group {x}" for x in unique]
    ax.bar(col_names, counts, color=['#e41a1c','#377eb8','#4daf4a'], alpha=0.8)
    #ax.set_xticks(col_names, fontsize = 14, weight = "bold")
    ax.set_title("(F) Neuron Group Counts",fontsize = 14, weight = "bold")

    # --- Plot 9: Decision Boundary for RNN Model ---
    ax = axs[1, 2]
    plot_decision_boundary(models['RNN'], X_train, y_train, "(G) Decision Boundary\n(RNN)", ax)

    # --- Plot 10: Decision Boundary for Heider-RNN Model ---
    ax = axs[1, 3]
    plot_decision_boundary(models['Heider-RNN'], X_train, y_train, "(H) Decision Boundary\n(Heider-RNN)", ax)

    plt.tight_layout()
    plt.savefig('result_2_rnn.jpg', dpi = 600)
    plt.show()



# ============================
# 5. Train Models and Generate Visualizations
# ============================
input_dim = 2
hidden_dim = 16
output_dim = 3
learning_rate = 0.01
num_epochs = 100

models = {
    'RNN': RNNModel(input_dim, hidden_dim, output_dim),
    'Heider-RNN': HeiderModel(input_dim, hidden_dim, output_dim, lambda_reg=0.01)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses, accs = train_model(model, criterion, optimizer, num_epochs)
    results[name] = (losses, accs)

'''# ============================
# 5. Visualization and Interpretation
# ============================
plt.figure(figsize=(14, 6))

# Loss comparison
plt.subplot(1, 2, 1)
for name, (losses, _) in results.items():
    plt.plot(losses, label=name)
plt.title('Training Loss Evolution')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy comparison
plt.subplot(1, 2, 2)
for name, (_, accs) in results.items():
    plt.plot(accs, label=name)
plt.title('Test Accuracy Evolution')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
'''


create_visualizations_with_decision_boundaries()
