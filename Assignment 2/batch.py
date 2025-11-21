import numpy as np
import matplotlib.pyplot as plt
import os

def h(x):
    """The target function we want the network to learn."""
    return 0.25 * x**2 * np.sin(x)

def generate_data(N):
    """
    Generates training and testing data sets based on the hyperparameter N.
    The training set has 8N+1 points, and the test set has 8N points.
    """
    num_train_points = 8 * N + 1
    x_train = np.linspace(-np.pi, np.pi, num_train_points).reshape(1, -1)
    d_train = h(x_train)
    x_test = (x_train[:, :-1] + x_train[:, 1:]) / 2
    d_test = h(x_test)
    return x_train.T, d_train.T, x_test.T, d_test.T

def tanh_derivative(x):
    """Derivative of the hyperbolic tangent function."""
    return 1 - np.tanh(x)**2

class NeuralNetwork:
    """
    A simple 2-layer neural network with a tanh hidden layer and linear output layer.
    """
    def __init__(self, n_hidden):
        """
        Initializes the network's weights and biases with small random values.
        """
        self.W1 = np.random.randn(n_hidden, 1) * 0.1
        self.b1 = np.random.randn(n_hidden, 1) * 0.1
        self.W2 = np.random.randn(1, n_hidden) * 0.1
        self.b2 = np.random.randn(1, 1) * 0.1

    def _forward_pass(self, x_sample):
        """
        Performs a forward pass for a single input sample.
        """
        v_hidden = self.W1 * x_sample - self.b1
        y_hidden = np.tanh(v_hidden)
        v_output = np.dot(self.W2, y_hidden) - self.b2
        y_output = v_output
        return v_hidden, y_hidden, y_output

    def predict(self, x_data):
        """
        Makes predictions for an entire dataset.
        """
        predictions = []
        for x_sample in x_data:
            y_output = self._forward_pass(x_sample)[2]
            predictions.append(y_output[0, 0])
        return np.array(predictions)

    def train(self, x_train, d_train, eta, error_threshold=1e-4, max_epochs=10000):
        """
        Trains the network using the Backpropagation algorithm (BATCH learning).
        """
        epoch_errors = []
        current_eta = eta
        num_samples = len(x_train)

        for epoch in range(max_epochs):
            # --- BATCH LEARNING MODIFICATION START ---

            # 1. Initialize gradient accumulators to zero for this epoch
            grad_W1 = np.zeros_like(self.W1)
            grad_b1 = np.zeros_like(self.b1)
            grad_W2 = np.zeros_like(self.W2)
            grad_b2 = np.zeros_like(self.b2)

            # 2. Loop through all samples to ACCUMULATE gradients
            for x_sample, d_sample in zip(x_train, d_train):
                # --- FORWARD PROPAGATION ---
                v_hidden, y_hidden, y_output = self._forward_pass(x_sample)

                # --- BACKWARD PROPAGATION ---
                error = d_sample - y_output
                
                # Calculate deltas (local gradients) for this one sample
                delta_output = error
                delta_hidden = (self.W2.T * delta_output) * tanh_derivative(v_hidden)
                
                # Add the calculated gradients for this sample to the accumulators
                grad_W2 += delta_output * y_hidden.T
                grad_b2 += delta_output * (-1.0)
                grad_W1 += delta_hidden * x_sample
                grad_b1 += delta_hidden * (-1.0)
            
            # 3. UPDATE weights and biases ONCE per epoch using the AVERAGE gradient
            self.W2 += current_eta * (grad_W2 / num_samples)
            self.b2 += current_eta * (grad_b2 / num_samples)
            self.W1 += current_eta * (grad_W1 / num_samples)
            self.b1 += current_eta * (grad_b1 / num_samples)
            
            # --- BATCH LEARNING MODIFICATION END ---
            
            # --- EVALUATE EPOCH (this part remains the same) ---
            y_pred_full = self.predict(x_train)
            current_E_avg = np.mean((d_train.flatten() - y_pred_full)**2)
            epoch_errors.append(current_E_avg)
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{max_epochs}, Average Error: {current_E_avg:.7f}")
            
            # --- APPLY LEARNING RATE DECAY ---
            if (epoch + 1) % 100 == 0 and epoch > 0:
                old_eta = current_eta
                current_eta *= 0.9
                print(f"  **Epoch {epoch+1}: Decaying learning rate from {old_eta:.5f} to {current_eta:.5f}**")

            # Check for convergence condition
            if current_E_avg < error_threshold:
                print(f"  Convergence achieved at epoch {epoch+1} with error {current_E_avg:.7f}")
                break
        
        return epoch_errors

# --- 3. MAIN EXECUTION SCRIPT --- (No changes needed below this line)

if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')
    print("'plots' directory has been created to store results.")

    n_values = [10, 15, 20]
    eta_values = [0.1, 0.3, 0.5]
    N_values = [20, 30, 40]

    results = []
    run_count = 1
    total_runs = len(n_values) * len(eta_values) * len(N_values)

    for N in N_values:
        for n in n_values:
            for eta in eta_values:
                print("-" * 60)
                print(f"Starting Run {run_count}/{total_runs}: (N={N}, n={n}, eta={eta})")
                
                x_train, d_train, x_test, d_test = generate_data(N)
                
                nn = NeuralNetwork(n_hidden=n)
                errors = nn.train(x_train, d_train, eta=eta)
                
                y_test = nn.predict(x_test)
                test_error = np.mean((d_test.flatten() - y_test)**2)
                
                result = {
                    'N': N, 'n': n, 'eta': eta,
                    'epochs': len(errors),
                    'final_train_error': errors[-1] if errors else float('inf'),
                    'test_error': test_error
                }
                results.append(result)
                print(f"  Training finished. Final Test Error (E_avg): {test_error:.7f}")
                print("  Now generating and saving plots...")

                run_id = f"N{N}_n{n}_eta{str(eta).replace('.', 'p')}_BATCH" # Added BATCH to filename
                
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(errors) + 1), errors, color='royalblue')
                plt.xlabel("Epochs")
                plt.ylabel("Average Training Error (E_ave)")
                plt.title(f"Training Error vs. Epochs (Batch Learning)\n(N={N}, n={n}, η={eta})")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.yscale('log')
                plt.savefig(os.path.join('plots', f'error_plot_{run_id}.png'))
                plt.close()

                plt.figure(figsize=(10, 6))
                x_smooth = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
                y_smooth_true = h(x_smooth)
                y_smooth_pred = nn.predict(x_smooth)
                
                plt.plot(x_smooth, y_smooth_true, label='True Function h(x)', color='blue', linewidth=2.5)
                plt.plot(x_smooth, y_smooth_pred, label='NN Output y(x)', color='red', linestyle='--', linewidth=2)
                plt.scatter(x_train, d_train, s=15, color='green', label='Training Points', alpha=0.7, zorder=5)
                plt.title(f"Function Approximation Result (Batch Learning)\n(N={N}, n={n}, η={eta})")
                plt.xlabel("x")
                plt.ylabel("h(x) / y(x)")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.ylim(-1.5, 1.5)
                plt.savefig(os.path.join('plots', f'approximation_plot_{run_id}.png'))
                plt.close()

                print(f"  Plots saved for {run_id}")
                run_count += 1

    print("\n" + "="*60)
    print("All experiments have been completed.")
    print("Plots are saved in the 'plots' directory.")

    print("\n" + "---" * 20)
    print("SUMMARY OF RESULTS (BATCH LEARNING)")
    print("---" * 20)
    print(f"{'N':<5}{'n':<5}{'eta':<7}{'Epochs':<8}{'Train Error':<15}{'Test Error':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['N']:<5}{res['n']:<5}{res['eta']:<7.3f}{res['epochs']:<8}"
              f"{res['final_train_error']:<15.7f}{res['test_error']:<15.7f}")