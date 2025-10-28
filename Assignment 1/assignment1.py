import numpy as np
import matplotlib.pyplot as plt

# Enable interactive mode for non-blocking plots during execution.
plt.ion()

class PerceptronTrainer:
    """
    A class to handle the single Perceptron training algorithm.
    It includes all steps from data generation to training and plotting.
    """
    def __init__(self, n: int, learning_rate: float, seed: int = 41)-> None:
        self.n = n
        self.eta = learning_rate
        self._rng = np.random.default_rng(seed=seed) # seeding is used for reproducibility.

        # Initialize weights [w1, w2, theta] as uniformly random float between -1 and 1s
        self.weights = self._rng.uniform(-1, 1, 3)
        
        # Generate n random points in the unit square [0, 1] x [0, 1]
        self.points = self._rng.random(size=(n, 2)) # This is the S that is mentioned in assignment pdf.

        # Create a random separating line and classify the points
        self.a = self._rng.random()
        self.b = self._rng.random()

        # Use the line passing from (0, a) and (1, b) to assign ground truth.
        self.labels = self._set_ground_truth()

    def _set_ground_truth(self) -> np.ndarray[float]:
        """
        Finds the truth values of the classes for the points according to the original separating line.
        The line passes through (0, a) and (1, b).
        Its equation is x_2 = (b-a)x_1 + a, which can be rewritten as (b-a)x1 - x2 + a = 0.
        
        - Class 1 (denoted by 1): Points are on or above the line ((b-a)x1 - x2 + a <= 0).
        - Class 2 (denoted by 0): Points are below the line.
        """
        labels = np.zeros(self.n)
        for i, p in enumerate(self.points):
            x1, x2 = p
            # Check the position of the point relative to the line
            if (self.b - self.a) * x1 - x2 + self.a <= 0:
                labels[i] = 1  # Class 1
            else:
                labels[i] = 0  # Class 2

        return labels

    def train(self, epoch_limit: int=300) -> tuple[int, list[int]]:
        """
        Trains the perceptron on the generated data until it converges. To prevent an infinite
        loop or extremely slow convergence, an epoch limit is used. Convergence is achieved when
        there are no misclassified points in an entire epoch.
        """
        epoch = 0
        misclassifications_history = []

        for epoch in range(epoch_limit):
            misclassified_count = 0
            # Iterate through all data points in each epoch
            for i in range(self.n):
                # For convenience, create a new numpy array by appending the 
                # threshold value -1 at the end of the points array.
                x = np.append(self.points[i] , [-1])
                # Apply the step activation function to get the output y
                y = 0 if x.dot(self.weights) < 0 else 1
                # Calculate the error (d - y) 
                error = self.labels[i] - y
                
                # If there is an error, update the weights and count the misclassification
                if error != 0:
                    misclassified_count += 1
                    # Perceptron update rule: w_new = w_old + eta * error * x
                    self.weights += self.eta * (error) * x

            misclassifications_history.append(misclassified_count)
            epoch += 1
            
            # The algorithm has converged if no points were misclassified in this epoch
            if misclassified_count == 0:
                break
        else:
            print(f"Warning: Reached epoch limit of {epoch_limit} without convergence.")

        return epoch, misclassifications_history

    def plot_results(self, epoch_history: list[int]) -> None:
        """
        Generates the two required plots after training is complete:
        A plot showing the data points, the original separating line,
           and the final separating line found by the perceptron.
        A plot of the epoch number vs. the number of misclassifications
           to visualize the convergence process.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left Plot (Data and Boundary Lines)
        ax1 = axes[0]
        class1_points = self.points[self.labels == 1]
        class2_points = self.points[self.labels == 0]
        ax1.scatter(class1_points[:, 0], class1_points[:, 1], c='red', marker='o', label='Class 1') 
        ax1.scatter(class2_points[:, 0], class2_points[:, 1], c='blue', marker='o', label='Class 0')

        # Plot the original boundary line: x2 = (b-a)x1 + a
        x_orig = np.array([0, 1])
        y_orig = (self.b - self.a) * x_orig + self.a
        ax1.plot(x_orig, y_orig, 'g--', linewidth=2, label='Original Boundary Line')

        # Plot the perceptron's final boundary line: Equation -> x2 = (theta - w1*x1) / w2
        w1, w2, theta = self.weights
        x_perc = np.array([0, 1])
        # Handle the case where w2 is close to zero (vertical line)
        if abs(w2) > 1e-6:
            y_perc = (theta - w1 * x_perc) / w2
            ax1.plot(x_perc, y_perc, 'm-', linewidth=2, label='Perceptron Final Line')
        else:
             # Plot a vertical line at x1 = theta/w1
            x_val = theta / w1 if abs(w1) > 1e-6 else 0.5
            ax1.axvline(x=x_val, color='m', linestyle='-', linewidth=2, label='Perceptron Final Line')

        ax1.set_title(f'Data and Boundary Lines', fontsize=16)
        ax1.set_xlabel('$x_1$', fontsize=12)
        ax1.set_ylabel('$x_2$', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='lower right')
        ax1.grid(True)
        ax1.set_aspect('equal', adjustable='box') # Make the plot a square

        # Right Plot (Convergence History)
        ax2 = axes[1]
        ax2.plot(range(1, len(epoch_history) + 1), epoch_history, marker='o', linestyle='-')
        ax2.set_title(f'Convergence Plot', fontsize=16)
        ax2.set_xlabel('Epoch Number', fontsize=12)
        ax2.set_ylabel('Number of Misclassified Points', fontsize=12)
        
        # Set integer x-ticks for epoch numbers
        if len(epoch_history) > 0:
            ax2.set_xticks(range(1, len(epoch_history) + 1))
        
        ax2.grid(True)
        
        # --- Final Figure Adjustments ---
        fig.suptitle(f'Perceptron Training Results (n={self.n}, η={self.eta})', fontsize=18)
        # Adjust layout to make space for suptitle and prevent overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        # Display the figure (non-blocking due to plt.ion() at the script start)
        plt.show() 
        # Pause to ensure the figure is drawn before the script prints the next block
        plt.pause(0.1) 


# Run simulation for n = 100 patterns 
print("\nRunning simulation for n = 100 patterns")

learning_rates = [0.1, 1, 10] # given in the assignment
for eta_val in learning_rates:
    print(f"\nTraining with n=100 and learning rate (η) = {eta_val}")
    
    # Create a PerceptronTrainer instance for the current configuration
    trainer_100 = PerceptronTrainer(n=100, learning_rate=eta_val)
    
    # Train the perceptron and get the history
    total_epochs, misclass_history = trainer_100.train()
    print(f" Convergence achieved in {total_epochs} epochs.")
    
    # Plot the results for this configuration
    trainer_100.plot_results(misclass_history)
    print(f"Plots for η = {eta_val} have been displayed.")

    # The equation for original line x2 = (b-a)x1 + a 
    print(f"\nThe original separating line for η = {eta_val}: x_2 = {trainer_100.a:.4f} + x_1({(trainer_100.b - trainer_100.a):.4f})")
    # The equation for perceptron's final line x2 = (theta - w1*x1) / w2
    print(f"The perceptron's final line for η = {eta_val}: x_2 = {(trainer_100.weights[2] / trainer_100.weights[1]):.4f} - x_1({(trainer_100.weights[0] / trainer_100.weights[1]):.4f})")

# Run simulation for n = 500 patterns 
print("Running simulation for n = 500 patterns")
for eta_val in learning_rates:
    print(f"\nTraining with n=500 and learning rate (η) = {eta_val}")
    
    # Create a PerceptronTrainer instance for the current configuration
    trainer_500 = PerceptronTrainer(n=500, learning_rate=eta_val)
    
    # Train the perceptron and get the history
    total_epochs, misclass_history = trainer_500.train()
    print(f"Convergence achieved in {total_epochs} epochs.")
    
    # Plot the results for this configuration
    trainer_500.plot_results(misclass_history)
    print(f"Plots for η = {eta_val} have been displayed.")

    # The equation for original line x2 = (b-a)x1 + a 
    print(f"\nThe original separating line for η = {eta_val}: x_2 = {trainer_500.a:.4f} + x_1({(trainer_500.b - trainer_500.a):.4f})")
    # The equation for perceptron's final line x2 = (theta - w1*x1) / w2
    print(f"The perceptron's final line for η = {eta_val}: x_2 = {(trainer_500.weights[2] / trainer_500.weights[1]):.4f} - x_1({(trainer_500.weights[0] / trainer_500.weights[1]):.4f})")

plt.show(block=True)