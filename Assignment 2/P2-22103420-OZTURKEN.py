import numpy as np
import matplotlib.pyplot as plt
import os

def h(x):
    """The target function we want the network to learn."""
    return 0.25 * x**2 * np.sin(x)

def generate_data(N):
    """
    Generates training and testing data sets based on the hyperparameter N.
    """
    # Generate 8N+1 training points evenly spaced from -pi to pi
    num_train_points = 8 * N + 1
    x_train = np.linspace(-np.pi, np.pi, num_train_points).reshape(1, -1)
    d_train = h(x_train)
    # Generate 8N test points, which are the midpoints of training points
    x_test = (x_train[:, :-1] + x_train[:, 1:]) / 2
    d_test = h(x_test)
    return x_train.T, d_train.T, x_test.T, d_test.T

def tanh_derivative(x):
    """Derivative of the hyperbolic tangent function."""
    return 1 - np.tanh(x)**2

class NeuralNetwork:
    """
    A 2-layer neural network with a tanh hidden layer and linear output layer.
    Can be trained using either 'online' or 'batch' learning.
    """
    def __init__(self, n_hidden):
        """
        Initializes the network's weights and biases with small random values.
        """
        # Layer 1 (Hidden Layer)
        self.W1 = np.random.randn(n_hidden, 1) * 0.1  # Weights from input to hidden
        self.b1 = np.random.randn(n_hidden, 1) * 0.1  # Biases for hidden layer
        # Layer 2 (Output Layer)
        self.W2 = np.random.randn(1, n_hidden) * 0.1  # Weights from hidden to output
        self.b2 = np.random.randn(1, 1) * 0.1  # Bias for output layer

    def _forward_pass(self, x_sample):
        """
        Performs a forward pass for a single input sample.
        """
        # Hidden Layer calculations
        v_hidden = self.W1 * x_sample - self.b1
        y_hidden = np.tanh(v_hidden) # Tanh activation
        # Output Layer calculations
        v_output = np.dot(self.W2, y_hidden) - self.b2
        y_output = v_output # Linear activation
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

    def train(self, x_train, d_train, eta, mode='online', error_threshold=1e-4, max_epochs=3000):
        """
        Trains the network using the Backpropagation algorithm.

        Args:
            x_train (np.array): Training input data.
            d_train (np.array): Training target data.
            eta (float): The learning rate.
            mode (str): The learning mode, either 'online' or 'batch'.
            error_threshold (float): The error target for convergence.
            max_epochs (int): The maximum number of epochs to train for.
        """
        epoch_errors = []
        current_eta = eta
        num_samples = len(x_train)

        for epoch in range(max_epochs):
            # --- CHOOSE LEARNING ALGORITHM BASED ON MODE ---
            if mode == 'online':
                # Online Learning: Update weights after each individual sample
                for x_sample, d_sample in zip(x_train, d_train):
                    v_hidden, y_hidden, y_output = self._forward_pass(x_sample)
                    error = d_sample - y_output # Loss derivative for linear output
                    # Calculate deltas (local gradients)
                    delta_output = error
                    delta_hidden = (self.W2.T * delta_output) * tanh_derivative(v_hidden)
                    
                    # Update weights and biases based on the deltas
                    self.W2 += current_eta * delta_output * y_hidden.T
                    self.b2 += current_eta * delta_output * (-1.0)
                    self.W1 += current_eta * delta_hidden * x_sample
                    self.b1 += current_eta * delta_hidden * (-1.0)

            elif mode == 'batch':
                # Batch Learning: Accumulate gradients and update once per epoch
                grad_W1, grad_b1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
                grad_W2, grad_b2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
                
                for x_sample, d_sample in zip(x_train, d_train):
                    # FORWARD PROPAGATION 
                    v_hidden, y_hidden, y_output = self._forward_pass(x_sample)
                    # BACKPROPAGATION
                    error = d_sample - y_output # Loss derivative for linear output

                    # Calculate deltas (local gradients) for this one sample
                    delta_output = error
                    delta_hidden = (self.W2.T * delta_output) * tanh_derivative(v_hidden)

                    # Add the calculated gradients for this sample to the accumulators
                    grad_W2 += delta_output * y_hidden.T
                    grad_b2 += delta_output * (-1.0)
                    grad_W1 += delta_hidden * x_sample
                    grad_b1 += delta_hidden * (-1.0)

                # Update weights with the average gradient
                self.W2 += current_eta * (grad_W2 / num_samples)
                self.b2 += current_eta * (grad_b2 / num_samples)
                self.W1 += current_eta * (grad_W1 / num_samples)
                self.b1 += current_eta * (grad_b1 / num_samples)
            
            else:
                raise ValueError("Invalid mode. Choose 'online' or 'batch'.")

            # CALCULATE AVERAGE ERROR FOR THE EPOCH
            y_pred_full = self.predict(x_train)
            current_E_avg = np.mean((d_train.flatten() - y_pred_full)**2)
            epoch_errors.append(current_E_avg)
            
            if (epoch + 1) % 1000 == 0: # Print less frequently for long runs
                print(f"  Epoch {epoch+1}/{max_epochs}, Average Error: {current_E_avg:.7f}")
            
            if (epoch + 1) % 100 == 0 and epoch > 0:
                old_eta = current_eta
                current_eta *= 0.9 # Decay the learning rate
                print(f"  **Epoch {epoch+1}: Decaying learning rate from {old_eta:.5f} to {current_eta:.5f}**")

            if current_E_avg < error_threshold:
                print(f"  Convergence achieved at epoch {epoch+1} with error {current_E_avg:.7f}")
                break
        
        return epoch_errors



if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')
    print("'plots' directory created to store results.")

    # Define the hyperparameter sets to iterate through
    n_values = [10, 15, 20] # Number of hidden neurons
    eta_values = [0.1, 0.3, 0.5] # Learning rates
    N_values = [20, 30, 40] # Training set sizes
    learning_modes = ['online', 'batch'] # Learning modes

    results = []
    total_runs = len(n_values) * len(eta_values) * len(N_values) * len(learning_modes)
    run_count = 1

    for mode in learning_modes:
        print("\n" + "="*60)
        print(f"--- STARTING EXPERIMENTS FOR: {mode.upper()} LEARNING ---")
        print("="*60)
        # Loop through all 27 combinations of hyperparameters
        for N in N_values:
            for n in n_values:
                for eta in eta_values:
                    print("-" * 60)
                    print(f"Starting Run {run_count}/{total_runs}: (Mode={mode}, N={N}, n={n}, eta={eta})")
                    
                    x_train, d_train, x_test, d_test = generate_data(N)
                    
                    nn = NeuralNetwork(n_hidden=n)
                    errors = nn.train(x_train, d_train, eta=eta, mode=mode)

                    # Evaluate the trained network on the test set
                    y_test = nn.predict(x_test)
                    test_error = np.mean((d_test.flatten() - y_test)**2)
                    
                    result = {
                        'mode': mode.title(), 'N': N, 'n': n, 'eta': eta,
                        'epochs': len(errors),
                        'final_train_error': errors[-1] if errors else float('inf'),
                        'test_error': test_error
                    }
                    results.append(result)
                    print(f"  Training finished. Final Test Error: {test_error:.7f}")

                    # PLOTTING
                    run_id = f"N{N}_n{n}_eta{str(eta).replace('.', 'p')}_{mode.upper()}"
                    
                    # Plot 1: Average Training Error vs. Epochs
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(errors) + 1), errors, color='royalblue')
                    plt.xlabel("Epochs")
                    plt.ylabel("Average Training Error (E_ave)")
                    plt.title(f"Training Error vs. Epochs ({mode.title()} Learning)\n(N={N}, n={n}, η={eta})")
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.yscale('log') # Log scale is essential for viewing error decrease
                    plt.savefig(os.path.join('plots', f'error_plot_{run_id}.png'))
                    plt.close()

                    # Plot 2: Learned Function vs. True Function
                    plt.figure(figsize=(10, 6))
                    x_smooth = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
                    y_smooth_true = h(x_smooth)
                    y_smooth_pred = nn.predict(x_smooth)
                    
                    plt.plot(x_smooth, y_smooth_true, label='True Function h(x)', color='blue', linewidth=2.5)
                    plt.plot(x_smooth, y_smooth_pred, label='NN Output y(x)', color='red', linestyle='--', linewidth=2)
                    plt.scatter(x_train, d_train, s=15, color='green', label='Training Points', alpha=0.7, zorder=5)
                    plt.title(f"Function Approximation ({mode.title()} Learning)\n(N={N}, n={n}, η={eta})")
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
    print("ALL EXPERIMENTS COMPLETED.")
    print("="*60)

    print("\n" + "---" * 20)
    print("SUMMARY OF RESULTS")
    print("---" * 20)
    print(f"{'Mode':<8}{'N':<5}{'n':<5}{'eta':<7}{'Epochs':<8}{'Train Error':<15}{'Test Error':<15}")
    print("-" * 70)
    for res in results:
        print(f"{res['mode']:<8}{res['N']:<5}{res['n']:<5}{res['eta']:<7.3f}{res['epochs']:<8}"
              f"{res['final_train_error']:<15.7f}{res['test_error']:<15.7f}")