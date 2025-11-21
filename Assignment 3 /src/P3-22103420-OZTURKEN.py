import numpy as np
import gzip
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    """
    Loads the MNIST dataset from the 'data' folder.
    """
    data_dir = 'data'
    print(f"Attempting to load MNIST data from: {data_dir}")

    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def load_images(filename):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
        return data

    def load_labels(filename):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data

    x_train = load_images(files['train_images'])
    y_train = load_labels(files['train_labels'])
    x_test = load_images(files['test_images'])
    y_test = load_labels(files['test_labels'])
    
    return (x_train, y_train), (x_test, y_test)


def encode_labels(labels):
    """
    Encodes integer labels into the target vector format [1, -1, ...].
    Digit '0' is mapped to the 10th neuron (index 9).
    """
    num_labels = len(labels)
    # Start with a matrix of all -1s
    encoded = np.full((num_labels, 10), -1.0)
    for i, label in enumerate(labels):
        # The project maps digit '1' to neuron 0, '2' to 1, ..., '0' to 9.
        if label == 0:
            encoded[i, 9] = 1.0
        else:
            encoded[i, label - 1] = 1.0
    return encoded


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_case=1):
        """
        Initializes the neural network's architecture and parameters.
        
        Args:
            input_size (int): Number of neurons in the input layer (784).
            hidden_size (int): Number of neurons in the hidden layer (e.g., 400 or 800).
            output_size (int): Number of neurons in the output layer (10).
            activation_case (int): 1 or 2, as defined in the project description.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_case = activation_case
        
        print(f"\nInitializing network...")
        print(f"Configuration: {hidden_size} hidden neurons, activation case {activation_case}")
        print("Weight initialization: Gaussian (mean=0, std_dev=0.1)")

        # --- Weight and Bias Initialization ---
        #Gaussian distribution with zero mean and a small variance is used.
        # The '+1' in the dimensions is for the bias/threshold term.
        std_dev = 0.1
        #Input to hidden weight initialization
        self.weights_ih = np.random.normal(0, std_dev, (self.hidden_size, self.input_size + 1))
        #Hidden to output weight initialization 
        self.weights_ho = np.random.normal(0, std_dev, (self.output_size, self.hidden_size + 1))
            
    # --- Activation Functions and their Derivatives ---

    def _tanh(self, v):
        return np.tanh(v)

    def _tanh_derivative(self, v):
        # Derivative of tanh(x) is 1 - tanh^2(x)
        return 1.0 - np.tanh(v)**2

    def _relu(self, v):
        return np.maximum(0, v)

    def _relu_derivative(self, v):
        # Derivative of ReLU is 1 for v > 0, and 0 otherwise
        return np.where(v > 0, 1, 0)
    
    def _apply_hidden_activation(self, v):
        if self.activation_case == 1: # Case 1: All tanh
            return self._tanh(v)
        elif self.activation_case == 2: # Case 2: Hidden is ReLU
            return self._relu(v)

    def _apply_hidden_activation_derivative(self, v):
        if self.activation_case == 1:
            return self._tanh_derivative(v)
        elif self.activation_case == 2:
            return self._relu_derivative(v)
            
    def _apply_output_activation(self, v):
        # Output activation is always tanh for both cases
        return self._tanh(v)

    def _apply_output_activation_derivative(self, v):
        # Derivative of output activation is always tanh derivative
        return self._tanh_derivative(v)

    def forward(self, inputs):
        """
        Performs a forward pass. Supports both single inputs and mini-batches.
        Input shape: (Batch_Size, 784) or (784,)
        Output shape: (10, Batch_Size) or (10,)
        """
        # Normalize input to be 2D: (Batch_Size, Input_Features)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            
        batch_size = inputs.shape[0]

        # --- Handle Bias ---
        # Add a column of 1s. Shape becomes (Batch_Size, 785)
        bias_column = np.ones((batch_size, 1))
        self.input_with_bias = np.concatenate((inputs, bias_column), axis=1)

        # --- Hidden Layer ---
        # weights_ih: (400, 785)
        # input_with_bias.T: (785, Batch_Size)
        # Result v_hidden: (400, Batch_Size)
        self.v_hidden = np.dot(self.weights_ih, self.input_with_bias.T)
        self.o_hidden = self._apply_hidden_activation(self.v_hidden)
        
        # Add bias row to hidden output
        # o_hidden: (400, Batch_Size) -> o_hidden_with_bias: (401, Batch_Size)
        bias_row = np.ones((1, batch_size))
        self.o_hidden_with_bias = np.concatenate((self.o_hidden, bias_row), axis=0)

        # --- Output Layer ---
        # weights_ho: (10, 401)
        # o_hidden_with_bias: (401, Batch_Size)
        # Result: (10, Batch_Size)
        self.v_output = np.dot(self.weights_ho, self.o_hidden_with_bias)
        self.o_output = self._apply_output_activation(self.v_output)

        # If original input was 1D, return 1D output
        if batch_size == 1:
            return self.o_output.flatten()
        return self.o_output

    def backward(self, target, learning_rate):
        """
        Performs backward pass with mini-batch support.
        Target shape: (Batch_Size, 10) or (10,)
        """
        # Ensure target is 2D: (Batch_Size, 10)
        target = np.array(target)
        if target.ndim == 1:
            target = target.reshape(1, -1)
            
        batch_size = target.shape[0]
        
        # Transpose target to match output shape (10, Batch_Size)
        target_T = target.T 

        # --- Calculate Error Deltas ---
        # error: (10, Batch_Size)
        error_output = target_T - self.o_output
        delta_output = error_output * self._apply_output_activation_derivative(self.v_output)
        
        # Backpropagate to hidden
        # weights_ho[:, :-1]: (10, 400) -> Transpose to (400, 10)
        # delta_output: (10, Batch_Size)
        # Result error_hidden: (400, Batch_Size)
        error_hidden = np.dot(self.weights_ho[:, :-1].T, delta_output)
        delta_hidden = error_hidden * self._apply_hidden_activation_derivative(self.v_hidden)

        # --- Update Weights ---
        # Update weights_ho
        # delta_output: (10, Batch_Size)
        # o_hidden_with_bias.T: (Batch_Size, 401)
        # Result: (10, 401)
        grad_ho = np.dot(delta_output, self.o_hidden_with_bias.T)
        self.weights_ho += (learning_rate / batch_size) * grad_ho

        # Update weights_ih
        # delta_hidden: (400, Batch_Size)
        # input_with_bias: (Batch_Size, 785)
        # Result: (400, 785)
        grad_ih = np.dot(delta_hidden, self.input_with_bias)
        self.weights_ih += (learning_rate / batch_size) * grad_ih
    

    def predict(self, inputs):
        """
        Predicts the digit. 
        """
        # If inputs are a single image 1D array
        output_activations = self.forward(inputs)
        
        # If batch prediction
        if output_activations.ndim == 2:
             # output shape (10, Batch) -> argmax over axis 0
            prediction_indices = np.argmax(output_activations, axis=0)
            # Handle 0 mapping
            return np.where(prediction_indices == 9, 0, prediction_indices + 1)
        
        # Single prediction
        prediction_index = np.argmax(output_activations)
        if prediction_index == 9:
            return 0
        else:
            return prediction_index + 1

    def evaluate(self, x_data, y_labels, y_encoded_data):
        """
        Evaluates the network's performance on a given dataset.
        
        Args:
            x_data: Input data (e.g., x_test).
            y_labels: The original numerical labels (e.g., test_labels).
            y_encoded_data: The one-hot encoded target vectors (e.g., y_test_encoded).

        Returns:
            A tuple of (average_error, success_rate).
        """
        correct_predictions = 0
        total_error = 0
        misclassified_counts = {i: 0 for i in range(10)}
        confusion_matrix = np.zeros((10, 10), dtype=int)
        
        inference_times = []
        
        for i in range(len(x_data)):
            start_time = time.perf_counter()
            prediction = self.predict(x_data[i]) 
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
            
            raw_output = self.forward(x_data[i])
            
            confusion_matrix[y_labels[i], prediction] += 1
            
            if prediction == y_labels[i]:
                correct_predictions += 1
            else:
              misclassified_counts[y_labels[i]] += 1
            
            error = 0.5 * np.sum((y_encoded_data[i] - raw_output)**2)
            total_error += error
            
        # Calculate final metrics
        average_error = total_error / len(x_data)
        success_rate = (correct_predictions / len(x_data)) * 100
        avg_inference_time = np.mean(inference_times)

        return {
            "avg_error": average_error,
            "success_rate": success_rate,
            "misclassified_per_class": misclassified_counts,
            "confusion_matrix": confusion_matrix,
            "avg_inference_time_ms": avg_inference_time * 1000
        }

    def train(self, x_train, y_labels, y_train_encoded, epochs, learning_rate, batch_size=1):
        """
        Mini-batch training loop. 
        If batch_size = 1, this performs Online (Stochastic) Learning.
        """
        if batch_size == 1:
            print(f"\n--- Starting Online Training ---")
            print(f"Epochs: {epochs}, Learning Rate (η): {learning_rate}")
        else:
            print(f"\n--- Starting Mini-batch Training ---")
            print(f"Epochs: {epochs}, Learning Rate (η): {learning_rate}, Batch Size: {batch_size}")
        
        epoch_times = []
        training_start_time = time.time()

        success_rates = []
        avg_train_errors = []
        num_samples = len(x_train)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_train_error = 0
            correct_predictions = 0
            
            # Shuffle indices
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_train_shuffled = x_train[indices]
            y_encoded_shuffled = y_train_encoded[indices]
            y_labels_shuffled = y_labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                
                # Create batch
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch_encoded = y_encoded_shuffled[start_idx:end_idx]
                y_batch_labels = y_labels_shuffled[start_idx:end_idx]
                
                # Forward
                # output shape: (10, Batch_Current_Size)
                output = self.forward(x_batch)
                
                # Backward
                self.backward(y_batch_encoded, learning_rate)
                
                # --- Metrics for this batch ---
                # Calculate batch error (Transposing target to match output shape (10, Batch))
                batch_error = 0.5 * np.sum((y_batch_encoded.T - output)**2)
                total_train_error += batch_error
                
                # Calculate accuracy
                # output: (10, Batch) -> argmax over axis 0
                preds = np.argmax(output, axis=0)
                # Map index 9 -> digit 0
                preds_digits = np.where(preds == 9, 0, preds + 1)
                
                correct_predictions += np.sum(preds_digits == y_batch_labels)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f}s")
            
            # --- Epoch Summary ---
            # Calculate average training error for the epoch
            avg_train_error = total_train_error / len(x_train)
            success_rate = (correct_predictions / len(x_train)) * 100
            print(f"Average Train Error: {avg_train_error} | Train Success Rate: {success_rate}")

            success_rates.append(success_rate)
            avg_train_errors.append(avg_train_error)

            
        training_end_time = time.time()
        return {
            "total_training_time": training_end_time - training_start_time,
            "avg_cpu_time_per_epoch": np.mean(epoch_times),
            "epochs_completed": epochs,
            "success_rates": success_rates,
            "avg_train_errors": avg_train_errors
        }


def print_results(config, training_stats, train_eval, test_eval):
    """
    Helper function to print the results in a structured format.
    """
    print("\n" + "="*80)
    print(f"               RESULTS FOR CONFIGURATION")
    print("="*80)

    if 'batch_size' in config:
        print(f"Hidden Neurons: {config['n_hidden']}, Activation Case: {config['act_case']}, Learning Rate: {config['lr']}, Batch Size: {config['batch_size']}")
    else:
        print(f"Hidden Neurons: {config['n_hidden']}, Activation Case: {config['act_case']}, Learning Rate: {config['lr']}")
    print("-"*80)
    print("\n--- TRAINING PERFORMANCE ---")
    print(f"Number of Epochs till Convergence: {training_stats['epochs_completed']}")
    print(f"Total Training Duration: {training_stats['total_training_time']:.2f} seconds")
    print(f"Average CPU Time per Epoch: {training_stats['avg_cpu_time_per_epoch']:.2f} seconds")
    print(f"Average Error (Training Set): {train_eval['avg_error']:.6f}")
    print(f"Success Rate (Training Set): {train_eval['success_rate']:.2f}%")
    print("Misclassified Patterns per Class (Training Set):")
    print(train_eval['misclassified_per_class'])
    
    print("\n--- TEST PERFORMANCE ---")
    print(f"Average Error (Test Set): {test_eval['avg_error']:.6f}")
    print(f"Success Rate (Test Set): {test_eval['success_rate']:.2f}%")
    print(f"Average Inference Time: {test_eval['avg_inference_time_ms']:.4f} ms per pattern")
    print("Misclassified Patterns per Class (Test Set):")
    print(test_eval['misclassified_per_class'])
    
    print("\n--- CONFUSION MATRIX (Test Set) ---")
    print("Rows: True Label, Columns: Predicted Label")
    print(test_eval['confusion_matrix'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_eval['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"Confusion Matrix (Test) - N={config['n_hidden']}, Act={config['act_case']}, LR={config['lr']}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("="*80 + "\n")


def run_online_experiments(x_train, train_labels, y_train_encoded, x_test, test_labels, y_test_encoded, num_epochs):
    print("\n\n" + "#"*30 + " STARTING ONLINE LEARNING EXPERIMENTS " + "#"*30)
    
    # Define hyperparameter combinations
    hidden_sizes = [400, 800]
    activation_cases = [1, 2] # 1: All-tanh, 2: Hidden-ReLU Output-tanh
    learning_rates = [0.01, 0.03]
    image_vector_size = 28*28

    for n_hidden in hidden_sizes:
        for act_case in activation_cases:
            for lr in learning_rates:
                
                config = {'n_hidden': n_hidden, 'act_case': act_case, 'lr': lr}
                
                print(f"\n--- Running new configuration: N={n_hidden}, Case={act_case}, η={lr} ---")
                
                # 1. Initialize the network
                nn = NeuralNetwork(
                    input_size=image_vector_size,
                    hidden_size=n_hidden,
                    output_size=10,
                    activation_case=act_case
                )
                
                # 2. Train the network (online learning)
                training_stats = nn.train(
                    x_train=x_train,
                    y_labels=train_labels,
                    y_train_encoded=y_train_encoded,
                    epochs=num_epochs,
                    learning_rate=lr
                )
                
                # 3. Evaluate on the training set
                print("Evaluating on Training Set...")
                train_eval_metrics = nn.evaluate(x_train, train_labels, y_train_encoded)
                
                # 4. Evaluate on the test set
                print("Evaluating on Test Set...")
                test_eval_metrics = nn.evaluate(x_test, test_labels, y_test_encoded)
                
                print_results(config, training_stats, train_eval_metrics, test_eval_metrics)



def run_minibatch_experiment(x_train, train_labels, y_train_encoded, x_test, test_labels, y_test_encoded, num_epochs, n_hidden, act_case, learning_rate):
    print("\n\n" + "#"*30 + " STARTING MINIBATCH LEARNING EXPERIMENT " + "#"*30)

    batch_sizes = [20, 200]
    image_vector_size = 28*28

    for batch_size in batch_sizes:
        config = {'n_hidden': n_hidden, 'act_case': act_case, 'lr': learning_rate, 'batch_size': batch_size}
        
        print(f"\n--- Running new configuration: N={n_hidden}, Case={act_case}, η={learning_rate}, Batch Size={batch_size} ---")
        
        # 1. Initialize the network
        nn = NeuralNetwork(
            input_size=image_vector_size,
            hidden_size=n_hidden,
            output_size=10,
            activation_case=act_case
        )
        
        # 2. Train the network (mini-batch learning)
        training_stats = nn.train(
            x_train=x_train,
            y_labels=train_labels,
            y_train_encoded=y_train_encoded,
            epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        # 3. Evaluate on the training set
        print("Evaluating on Training Set...")
        train_eval_metrics = nn.evaluate(x_train, train_labels, y_train_encoded)
        
        # 4. Evaluate on the test set
        print("Evaluating on Test Set...")
        test_eval_metrics = nn.evaluate(x_test, test_labels, y_test_encoded)
        
        print_results(config, training_stats, train_eval_metrics, test_eval_metrics)

if __name__ == "__main__":    
    try:
        (train_images, train_labels), (test_images, test_labels) = load_data()
        # Print the shape of the loaded data to verify
        print("\nSuccessfully loaded the data.")
        print("Training images shape:", train_images.shape)
        print("Training labels shape:", train_labels.shape)
        print("Test images shape:", test_images.shape)
        print("Test labels shape:", test_labels.shape)

    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"Error: {e}")
        print("\nCould not find the data files. Please ensure the 'data' folder exists and contains the MNIST files.")
        exit()

    # 1. Flatten the images
    image_vector_size = 28 * 28
    x_train = train_images.reshape(train_images.shape[0], image_vector_size).astype('float32')
    x_test = test_images.reshape(test_images.shape[0], image_vector_size).astype('float32')

    print("--- After Flattening ---")
    print("Training images shape:", x_train.shape)
    print("Test images shape:", x_test.shape)

    # 2. Normalize the pixel values
    x_train /= 255.0
    x_test /= 255.0
    print("\n--- After Normalization ---")
    print("Max pixel value in training set:", np.max(x_train))
    print("Min pixel value in training set:", np.min(x_train))

    y_train_encoded = encode_labels(train_labels)
    y_test_encoded = encode_labels(test_labels)

    print("\n--- After Encoding Labels ---")
    print("Encoded training labels shape:", y_train_encoded.shape)
    print("Encoded test labels shape:", y_test_encoded.shape)

    # Let's check an example
    print("\nExample:")
    print("Original first label:", train_labels[0])
    print("Encoded first label:", y_train_encoded[0]) 

    original_label_for_0 = np.where(train_labels == 0)[0][0]
    print("\nAn example for digit 0:")
    print("Original label:", train_labels[original_label_for_0])
    print("Encoded label for 0:", y_train_encoded[original_label_for_0])

    NUM_EPOCHS = 50

    # Online Learning Experiments
    run_online_experiments(x_train, train_labels, y_train_encoded, x_test, test_labels, y_test_encoded, NUM_EPOCHS)

    # Best Parameters found after 8 hyperparameter combinations results in online learning
    n_hidden = 800
    act_case = 2
    learning_rate = 0.01

    run_minibatch_experiment(x_train, train_labels, y_train_encoded, x_test, test_labels, y_test_encoded, NUM_EPOCHS, n_hidden, act_case, learning_rate)



