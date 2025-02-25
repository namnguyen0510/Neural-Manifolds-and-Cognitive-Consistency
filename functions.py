import numpy as np




# ----------------------------
# 1. Helper Functions
# ----------------------------
def compute_balance_energy(W):
    """
    Compute the balance energy of the network:
    E_balance = sum_{i<j<k} (W[i,j]*W[j,k]*W[k,i] + 1)^2.
    """
    N = W.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                term = W[i, j] * W[j, k] * W[k, i] + 1
                energy += term**2
    return energy

def compute_balance_gradient(W):
    """
    Compute the gradient of the balance energy with respect to W.
    For each triad (i,j,k) with i<j<k, the gradient contributions are:
      - dE/dW[i,j] += 2*(W[i,j]*W[j,k]*W[k,i] + 1) * (W[j,k]*W[k,i])
      - dE/dW[j,k] += 2*(W[i,j]*W[j,k]*W[k,i] + 1) * (W[i,j]*W[k,i])
      - dE/dW[k,i] += 2*(W[i,j]*W[j,k]*W[k,i] + 1) * (W[i,j]*W[j,k])
    (Then we symmetrize the gradient.)
    """
    N = W.shape[0]
    grad = np.zeros_like(W)
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                triad_val = W[i, j] * W[j, k] * W[k, i] + 1
                factor = 2 * triad_val
                # Contribution for W[i,j]
                grad[i, j] += factor * (W[j, k] * W[k, i])
                grad[j, i] += factor * (W[j, k] * W[k, i])
                # Contribution for W[j,k]
                grad[j, k] += factor * (W[i, j] * W[k, i])
                grad[k, j] += factor * (W[i, j] * W[k, i])
                # Contribution for W[k,i]
                grad[k, i] += factor * (W[i, j] * W[j, k])
                grad[i, k] += factor * (W[i, j] * W[j, k])
    return grad

def compute_balance_energy_fast(W):
    """
    Fast computation of the balance energy using matrix trace identities.
    For a symmetric matrix W:
      E_balance = C(N,3) + trace(W^3)/3 + trace((W ∘ W)^3)/6,
    where (W ∘ W) denotes elementwise square of W.
    # Example usage:
        N = 10
        W = np.random.uniform(-1, 1, (N, N))
        W = (W + W.T) / 2  # enforce symmetry
        np.fill_diagonal(W, 0)

        energy_fast = compute_balance_energy_fast(W)
        print("Fast Energy:", energy_fast)
    """
    N = W.shape[0]
    comb = N * (N - 1) * (N - 2) / 6  # Number of triads
    energy = comb + np.trace(W @ W @ W) / 3 + np.trace((np.square(W) @ np.square(W) @ np.square(W))) / 6
    return energy

def compute_balance_gradient_fast(W):
    """
    Fast computation of the gradient of the balance energy with respect to W,
    using the closed-form expression derived from trace identities.
    
    Returns:
      grad_fast = 2 * (W @ W + (np.dot(np.square(W), np.square(W)) * W)),
      where the second term is computed via a matrix product followed by an elementwise multiplication with W.
    # Example usage:
        grad_fast = compute_balance_gradient_fast(W)
        print("Fast Gradient:\n", grad_fast)
    """
    term1 = W @ W
    term2 = np.dot(np.square(W), np.square(W)) * W  # elementwise multiplication after matrix product
    grad_fast = 2 * (term1 + term2)
    return grad_fast

def enforce_symmetry(W):
    """
    Ensure that the weight matrix remains symmetric and has zero diagonal.
    """
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    return W