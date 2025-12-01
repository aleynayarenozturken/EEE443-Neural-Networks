import numpy as np
import sys
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

(train_images, train_labels), (test_images, test_labels) = load_data()
print("MNIST data loaded successfully.")

#Data Preprocessing
#Flatten the images
image_vector_size = 28 * 28
x_train = train_images.reshape(train_images.shape[0], image_vector_size).astype('float32')
x_test = test_images.reshape(test_images.shape[0], image_vector_size).astype('float32')

print("--- After Flattening ---")
print("Training images shape:", x_train.shape)
print("Test images shape:", x_test.shape)

#Normalize the pixel values
x_train /= 255.0
x_test /= 255.0
print("\n--- After Normalization ---")
print("Max pixel value in training set:", np.max(x_train))
print("Min pixel value in training set:", np.min(x_train))

def encode_labels(y, num_classes=10):
    """
    One-hot encodes the labels.
    Example: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    """
    one_hot = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        one_hot[i, y[i]] = 1.0
    return one_hot


class NeuralNetwork:
    def __init__(self, use_threshold=True, activation_type='sigmoid'):
        self.use_threshold = use_threshold
        self.activation_type = activation_type
        self.parameters = {}
        np.random.seed(42)
        
        def init_w(shape): return np.random.normal(0, 0.1, size=shape)
        def init_b(shape): return np.random.normal(0, 0.1, size=shape) if use_threshold else np.zeros(shape)

        self.parameters['W1'] = init_w((4, 5, 5))
        self.parameters['b1'] = init_b((4, 1))
        self.parameters['W3'] = init_w((3, 5, 5))
        self.parameters['b3'] = init_b((3, 1))
        self.parameters['W5'] = init_w((10, 192))
        self.parameters['b5'] = init_b((10, 1))

    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def _im2col(self, img, kernel_size):
        """
        Transforms a 2D image into a column matrix where each column is a flattened 
        kernel-sized patch, facilitating matrix multiplication for convolution.
        """
        h, w = img.shape
        k = kernel_size
        oh, ow = h - k + 1, w - k + 1
        
        # Create indices for the input image using broadcasting/tiling/repeating
        i0 = np.repeat(np.arange(k), k)
        i1 = np.repeat(np.arange(oh), ow)
        j0 = np.tile(np.arange(k), k)
        j1 = np.tile(np.arange(ow), oh)
        
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        
        # Extract columns
        cols = img[i, j]
        return cols
    
    def _convolve_2d(self, img, kern, bias):
        """
        Vectorized 2D Convolution using im2col and dot product.
        Input: img (H, W), kern (K, K), bias (1,)
        Output: out (OH, OW)
        """
        k = kern.shape[0]
        oh, ow = img.shape[0] - k + 1, img.shape[1] - k + 1
        
        #Im2Col Transformation
        img_cols = self._im2col(img, k) # Shape: (k*k, oh*ow)
        
        #Kernel Reshape
        kern_row = kern.reshape(1, -1) # Shape: (1, k*k)
        
        #Matrix Multiplication (The core efficiency gain)
        # (1, k*k) @ (k*k, oh*ow) -> (1, oh*ow)
        out_flat = np.dot(kern_row, img_cols) 
        
        #Reshape and Add Bias
        b_val = float(bias) if np.isscalar(bias) else float(bias[0])
        out = (out_flat + b_val).reshape(oh, ow)
        return out

    def _pool_2d(self, img):
        """
        Vectorized Mean Pooling (Kernel=2, Stride=2).
        """
        h, w = img.shape
        k = 2 # Kernel size
        
        # Reshape the image to group 2x2 blocks together: (H/k, k, W/k, k)
        # Then take the mean over the two intermediate axes (1 and 3).
        pooled = img.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
        
        return pooled

    def _upsample_error(self, err):
        # Backward pass for 2x2 Average Pooling: distributes the error equally to the 2x2 input block.
        return np.repeat(np.repeat(err, 2, axis=0), 2, axis=1) / 4.0

    def _convolve_backprop(self, err, kern):
        pad = kern.shape[0] - 1
        padded = np.pad(err, ((pad,pad),(pad,pad)), mode='constant')
        return self._convolve_2d(padded, np.rot90(kern, 2), 0)


    def forward(self, img_flat):
        params = self.parameters
        img = img_flat.reshape(28, 28)
        
        # H1 & H2
        H1 = np.array([self._convolve_2d(img, params['W1'][f], params['b1'][f]) for f in range(4)])
        H2 = np.array([self._pool_2d(m) for m in H1])
        
        # H3
        H3 = []
        map_info = [] 
        for i in range(4):
            for k in range(3):
                H3.append(self._convolve_2d(H2[i], params['W3'][k], params['b3'][k]))
                map_info.append((i, k))
        H3 = np.array(H3)
        
        # H4
        H4 = np.array([self._pool_2d(m) for m in H3])
        H4_flat = H4.reshape(-1, 1)
        
        # H5
        Z5 = np.dot(params['W5'], H4_flat) + params['b5']
        
        if self.activation_type == 'sigmoid': output = self._sigmoid(Z5)
        else: output = self._softmax(Z5)
            
        return output, {'i': img, 'h1': H1, 'h2': H2, 'h3': H3, 'h4': H4, 'map_info': map_info, 'h4_flat': H4_flat, 'out': output}

    def backward(self, cache, target):
        p = self.parameters
        out, target = cache['out'], target.reshape(10, 1)
        
        if self.activation_type == 'sigmoid':
            err = (out - target)
            d5 = err * out * (1 - out)
            loss = 0.5 * np.sum(err ** 2)
        else:
            d5 = out - target
            loss = -np.sum(target * np.log(out + 1e-9))

        grads = {}
        grads['W5'] = np.dot(d5, cache['h4_flat'].T)
        grads['b5'] = d5
        
        d4_flat = np.dot(p['W5'].T, d5)
        d4_maps = d4_flat.reshape(12, 4, 4)
        d3_maps = np.array([self._upsample_error(m) for m in d4_maps])
        
        grads['W3'], grads['b3'] = np.zeros_like(p['W3']), np.zeros_like(p['b3'])
        d2_maps = np.zeros_like(cache['h2'])
        
        for m in range(12):
            in_idx, filt_idx = cache['map_info'][m]
            curr_delta = d3_maps[m]
            grads['W3'][filt_idx] += self._convolve_2d(cache['h2'][in_idx], curr_delta, 0)
            grads['b3'][filt_idx] += np.sum(curr_delta)
            d2_maps[in_idx] += self._convolve_backprop(curr_delta, p['W3'][filt_idx])

        d1_maps = np.array([self._upsample_error(m) for m in d2_maps])
        grads['W1'], grads['b1'] = np.zeros_like(p['W1']), np.zeros_like(p['b1'])
        for f in range(4):
            grads['W1'][f] = self._convolve_2d(cache['i'], d1_maps[f], 0)
            grads['b1'][f] = np.sum(d1_maps[f])

        return grads, loss

    def _update_params(self, grads, lr):
        for k in self.parameters:
            self.parameters[k] -= lr * grads[k]

    def predict(self, image_flat):
        out, _ = self.forward(image_flat)
        return np.argmax(out)

    def evaluate(self, x_data, y_data):
        correct, total_loss = 0, 0
        N = len(x_data)
        for i in range(N):
            out, _ = self.forward(x_data[i])
            if self.activation_type == 'sigmoid':
                total_loss += 0.5 * np.sum((out.flatten() - y_data[i])**2)
            else:
                total_loss += -np.sum(y_data[i] * np.log(out.flatten() + 1e-9))
            if np.argmax(out) == np.argmax(y_data[i]): correct += 1
        return correct / N, total_loss / N

    def train(self, x_train, y_train, epochs=5, lr=0.01, batch_size=1):
        """
        Trains the neural network using specified batch size.
        """
        mode = "Online" if batch_size == 1 else f"Batch(B={batch_size})"
        print(f"--- Training {self.activation_type} | {mode} | lr={lr} ---")
        
        # History only tracks Training Metrics
        history = {'loss': [], 'acc': [], 'epoch_times': []}
        N = len(x_train)
        
        total_start = time.time()
        for ep in range(epochs):
            ep_start = time.time()
            ep_loss, correct = 0, 0
            indices = np.arange(N)
            np.random.shuffle(indices)
            
            accum_grads = {k: np.zeros_like(v) for k, v in self.parameters.items()}
            batch_count = 0
            
            for i in indices:
                out, cache = self.forward(x_train[i])
                grads, loss = self.backward(cache, y_train[i])
                ep_loss += loss
                if np.argmax(out) == np.argmax(y_train[i]): correct += 1
                
                for k in grads: accum_grads[k] += grads[k]
                batch_count += 1
                
                if batch_count == batch_size:
                    avg_grads = {k: v / batch_size for k, v in accum_grads.items()}
                    self._update_params(avg_grads, lr)
                    for k in accum_grads: accum_grads[k].fill(0.0)
                    batch_count = 0
            
            if batch_count > 0:
                avg_grads = {k: v / batch_count for k, v in accum_grads.items()}
                self._update_params(avg_grads, lr)

            ep_duration = time.time() - ep_start
            history['epoch_times'].append(ep_duration)
            train_acc = correct / N
            
            history['loss'].append(ep_loss/N)
            history['acc'].append(train_acc)
            
            
            print(f"Epoch {ep+1}: Loss={ep_loss/N:.4f}, Train Acc={train_acc:.1%} ({ep_duration:.1f}s)")
            
        history['total_duration'] = time.time() - total_start
        history['avg_epoch_time'] = sum(history['epoch_times']) / len(history['epoch_times'])
        history['epochs_run'] = epochs
        return history


def generate_report(network, history, x_train, y_train, x_test, y_test, config, visualize_idx=0):
    """
    Generates a detailed report of the network's performance including:
    1. Training and Test Statistics
    2. Confusion Matrix
    3. Layer Visualizations for a specific test image
    """

    #Training Stats
    train_misclassified = np.zeros(10, dtype=int)
    
    #Predict on training set
    limit_check = min(len(x_train), 1000)
    for i in range(limit_check):
        pred_idx = network.predict(x_train[i])
        true_idx = np.argmax(y_train[i])
        
        if pred_idx != true_idx:
            train_misclassified[true_idx] += 1

    #Test Stats & Confusion Matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    test_misclassified = np.zeros(10, dtype=int)
    
    start_inf = time.time()
    for i in range(len(x_test)):
        pred_idx = network.predict(x_test[i])
        true_idx = np.argmax(y_test[i])
        
        # Populate Matrix (Row=True, Col=Pred)
        confusion_matrix[true_idx, pred_idx] += 1
        
        if pred_idx != true_idx:
            test_misclassified[true_idx] += 1
            
    end_inf = time.time()
    avg_inference_ms = ((end_inf - start_inf) / len(x_test)) * 1000

    #Print report
    print("\n" + "="*80)
    print(f"               RESULTS FOR CONFIGURATION: {config.get('name', 'Custom')}")
    print("="*80)

    print(f"Hidden Neurons: LeNet-5, Activation: {network.activation_type}, "
          f"LR: {config.get('lr', 0.01)}, Batch: {config.get('batch_size', 1)}")
    print("-"*80)
    
    print("\n--- TRAINING PERFORMANCE ---")
    print(f"Epochs Run: {history.get('epochs_run', len(history['loss']))}")
    print(f"Total Duration: {history.get('total_duration', 0):.2f} s")
    print(f"Final Train Error: {history['loss'][-1]:.6f}")
    print(f"Final Train Accuracy: {history['acc'][-1]*100:.2f}%")
    print("Misclassified Patterns per Class (Training Set):")
    print(train_misclassified.tolist())
    
    print("\n--- TEST PERFORMANCE ---")
    test_acc, test_err = network.evaluate(x_test, y_test)
    print(f"Final Test Error: {test_err:.6f}")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Avg Inference Time: {avg_inference_ms:.4f} ms per pattern")
    print("Misclassified Patterns per Class (Test Set):")
    print(test_misclassified.tolist())
    
    print("\n--- CONFUSION MATRIX (Test Set) ---")
    print("Rows: True Digit (0-9), Columns: Predicted Digit (0-9)")
    print(confusion_matrix)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"Confusion Matrix - {config.get('name', '')}")
    plt.ylabel('True Digit')
    plt.xlabel('Predicted Digit')
    plt.show()

    print("="*80 + "\n")

    #Layer Visualizations
    print(f"\n>>> VISUALIZING LAYERS FOR TEST IMAGE INDEX: {visualize_idx}")
    
    img_vis = x_test[visualize_idx]
    
    # Forward pass
    pred_vec, cache = network.forward(img_vis)
    
    # Determine True and Predicted Digits
    true_digit = np.argmax(y_test[visualize_idx])
    pred_digit = np.argmax(pred_vec)
    
    print(f"True Digit: {true_digit} | Predicted Digit: {pred_digit}")
    
    # Plot Input
    plt.figure(figsize=(2, 2))
    plt.imshow(img_vis.reshape(28, 28), cmap='gray')
    plt.title(f"Input: {true_digit}")
    plt.axis('off')
    plt.show()
    
    # Plot Layers
    layer_names = [('h1', 'H1 (Conv 5x5)', 4), 
                   ('h2', 'H2 (Pool 2x2)', 4), 
                   ('h3', 'H3 (Conv 5x5)', 12), 
                   ('h4', 'H4 (Pool 2x2)', 12)]
                   
    for key, title, n_maps in layer_names:
        print(f"Layer {title}:")
        rows = 1 if n_maps <= 4 else 2
        cols = n_maps if n_maps <= 4 else 6
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        if n_maps > 1: axes = axes.flatten()
        else: axes = [axes]
        
        for i in range(n_maps):
            axes[i].imshow(cache[key][i], cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    
    print("="*80)


def plot_case_history(history, title):
    """
    Plots Training Loss and Training Accuracy in figures.
    """
    epochs_range = range(1, len(history['loss']) + 1)
    
    #Training Loss Figure
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['loss'], 'r-o', linewidth=2, label='Training Loss')
    plt.title(f"{title}: Training Loss per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs_range)  # Force integer ticks
    plt.grid(True)
    plt.legend()
    plt.show()
    
    #Training Accuracy Figure
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['acc'], 'b-o', linewidth=2, label='Training Accuracy')
    plt.title(f"{title}: Training Accuracy per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(epochs_range)  # Force integer ticks
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":    
    try:
        (train_images, train_labels), (test_images, test_labels) = load_data()
        print("\nSuccessfully loaded the data.")
    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"Error: {e}")
        print("\nCould not find the data files. Please ensure the 'data' folder exists and contains the MNIST files.")
        sys.exit() # Exit if no data

    
    image_vector_size = 28 * 28
    
    
    x_train_all = train_images.reshape(train_images.shape[0], image_vector_size).astype('float32') / 255.0
    x_test_all = test_images.reshape(test_images.shape[0], image_vector_size).astype('float32') / 255.0
    
    
    y_train_encoded_all = encode_labels(train_labels)
    y_test_encoded_all = encode_labels(test_labels)

            
    EPOCHS = 30         
    LEARNING_RATE = 0.01
    
    
    x_train = x_train_all
    y_train = y_train_encoded_all
    x_test = x_test_all
    y_test = y_test_encoded_all
    
    print(f"\n>>> CONFIGURATION: N_Train={len(x_train)}, Epochs={EPOCHS}, LR={LEARNING_RATE}")

    #Running Online Cases
    print("\n" + "="*60)
    print(">>>RUNNING ONLINE CASES")
    print("="*60)

    cases = [
        {'name': 'Case 1 (Sigmoid, Thresh=True)',  'act': 'sigmoid', 'thresh': True,  'vis_idx': 0},
        {'name': 'Case 2 (Sigmoid, Thresh=False)', 'act': 'sigmoid', 'thresh': False, 'vis_idx': 1},
        {'name': 'Case 3 (Softmax, Thresh=True)',  'act': 'softmax', 'thresh': True,  'vis_idx': 8},
        {'name': 'Case 4 (Softmax, Thresh=False)', 'act': 'softmax', 'thresh': False, 'vis_idx': 16},
    ]

    online_results = {}
    best_acc = -1
    best_case_config = None

    for case in cases:
        print(f"\n>>> STARTED: {case['name']}...")
        
        #Initialize & Train
        nn = NeuralNetwork(use_threshold=case['thresh'], activation_type=case['act'])
        history = nn.train(x_train, y_train, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=1)
        
        #Plot Graphs 
        plot_case_history(history, case['name'])

        #Generate Report
        config_dict = {'name': case['name'], 'act_case': case['act'], 'lr': LEARNING_RATE, 'batch_size': 1}
        generate_report(nn, history, x_train, y_train, x_test, y_test, 
                        config=config_dict, visualize_idx=case['vis_idx'])
        
        #Calculate final test accuracy strictly for Winner Determination
        final_test_acc, _ = nn.evaluate(x_test, y_test)
        
        #Store results
        history['final_test_acc'] = final_test_acc
        online_results[case['name']] = history
        
        if final_test_acc > best_acc:
            best_acc = final_test_acc
            best_case_config = case

    print("\n" + "*"*60)
    print(f" WINNER: {best_case_config['name']} ({best_acc*100:.2f}%)")
    print("*"*60)

    winner_name = best_case_config['name']
    winner_act = best_case_config['act']
    winner_thresh = best_case_config['thresh']
    print(f"\n>>>RUNNING BATCH EXPERIMENTS ON {winner_name}")
    print("="*60)

    batch_sizes = [10, 100]
    batch_results = {}

    for b_size in batch_sizes:
        title = f"Batch Size {b_size} ({winner_name})"
        print(f"\n>>> STARTED: {title}...")
        
        #Initialize Winner Network (Sigmoid, Threshold=True)
        nn_batch = NeuralNetwork(use_threshold=winner_thresh, 
                                    activation_type=winner_act)
        
        #Train (Batch Mode)
        history = nn_batch.train(x_train, y_train, epochs= 50, lr=LEARNING_RATE, batch_size=b_size)
        
        #Plot Training Loss and Accuracy specific to this Batch Run
        plot_case_history(history, title)
        
        #Generate Full Report (Confusion Matrix, Inference Time, Visualization, etc.)
        config_dict = {
            'name': title, 
            'act_case': winner_act, 
            'lr': LEARNING_RATE, 
            'batch_size': b_size
        }
        
        generate_report(nn_batch, history, x_train, y_train, x_test, y_test, 
                        config=config_dict, visualize_idx= 0)
            


    
   