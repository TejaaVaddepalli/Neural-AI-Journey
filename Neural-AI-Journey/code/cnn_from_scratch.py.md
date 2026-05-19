import numpy as np

import matplotlib.pyplot as plt

  

# --- OPTIMIZED CONVOLUTION WITH MOMENTUM ---

class ConvolutionLeyrer:

    def __init__(self, in_channels, num_kernels, kernel_size, beta=0.9):

        self.num_kernel = num_kernels

        self.f_size = kernel_size

        self.beta = beta

        n_in = kernel_size * kernel_size * in_channels

        self.W = np.random.randn(kernel_size, kernel_size, in_channels, num_kernels) * np.sqrt(2/n_in)

        self.b = np.zeros(num_kernels)

        # MOMENTUM INITIALIZATION (The Memory)

        self.vW = np.zeros_like(self.W)

        self.vb = np.zeros_like(self.b)

    def forward(self, X):

        self.X_input = X

        batch, h_in, w_in, c_in = X.shape

        X_padded = np.pad(X, ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant')

        self.X_padded = X_padded

        Z = np.zeros((batch, h_in, w_in, self.num_kernel))

  

        for i in range(self.f_size):

            for j in range(self.f_size):

                X_slice = X_padded[:, i:i+h_in, j:j+w_in, :]

                for f in range(self.num_kernel):

                    Z[:, :, :, f] += np.sum(X_slice * self.W[i, j, :, f], axis=-1)

        Z += self.b

        self.A = np.where(Z > 0, Z, Z * 0.01) # Leaky ReLU

        return self.A

  

    def backward(self, dZ_in, learning_rate):

        dZ = dZ_in * np.where(self.A > 0, 1, 0.01)

        batch, h_out, w_out, num_f = dZ.shape

        dW = np.zeros_like(self.W)

        db = np.zeros_like(self.b)

        dX_padded = np.zeros_like(self.X_padded)

  

        for i in range(self.f_size):

            for j in range(self.f_size):

                X_slice = self.X_padded[:, i:i+h_out, j:j+w_out, :]

                for f in range(num_f):

                    # Error Projection with 5D Axis expansion

                    dZ_5D = dZ[:, :, :, f, np.newaxis]

                    dW[i, j, :, f] = np.sum(X_slice * dZ_5D, axis=(0, 1, 2))

                    dX_padded[:, i:i+h_out, j:j+w_out, :] += self.W[i, j, :, f] * dZ_5D

        db = np.sum(dZ, axis=(0, 1, 2))

  

        # MOMENTUM UPDATE (The Velocity)

        self.vW = self.beta * self.vW + (1 - self.beta) * (dW / batch)

        self.vb = self.beta * self.vb + (1 - self.beta) * (db / batch)

        self.W -= learning_rate * self.vW

        self.b -= learning_rate * self.vb

        return np.clip(dX_padded[:, 1:-1, 1:-1, :], -1, 1)

class MaxPoolingLayer:

    def __init__(self, pool_size=2, stride=2):

        self.p_size = pool_size

        self.stride = stride

  

    def forward(self, X):

        self.X_input = X

        b, h, w, c = X.shape

        # VECTORIZED MAX POOL: Reshape and find max

        X_reshaped = X.reshape(b, h // self.p_size, self.p_size, w // self.p_size, self.p_size, c)

        out = X_reshaped.max(axis=(2, 4))

        # Mask for backward (where the max was)

        self.mask = (X_reshaped == out[:, :, np.newaxis, :, np.newaxis, :])

        return out

  

    def backward(self, dZ):

        b, h, w, c = self.X_input.shape

        # VECTORIZED BACKWARD: Expand dZ to match the reshaped mask

        dZ_expanded = dZ[:, :, np.newaxis, :, np.newaxis, :]

        dX_reshaped = dZ_expanded * self.mask

        return dX_reshaped.reshape(b, h, w, c)

class FlattenLayer:

    def forward(self,X):

        self.input_shape = X.shape # (Batch, 14, 14, 8)

        # Flatten everything except the Batch

        return X.reshape(X.shape[0], -1)

    def backward(self, dZ):

        # Turn the 1D line of error back into the 3D block

        return dZ.reshape(self.input_shape)

# --- OPTIMIZED DENSE WITH MOMENTUM ---

class DenseLayer:

    def __init__(self, input_size, output_size, beta=0.9):

        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)

        self.b = np.zeros((1, output_size))

        self.beta = beta

        self.vW = np.zeros_like(self.W)

        self.vb = np.zeros_like(self.b)

  

    def forward(self, X):

        self.X_input = X

        return np.dot(X, self.W) + self.b

  

    def backward(self, dZ, learning_rate):

        batch = self.X_input.shape[0]

        dW = np.dot(self.X_input.T, dZ) / batch

        db = np.sum(dZ, axis=0, keepdims=True) / batch

        dX = np.dot(dZ, self.W.T)

  

        # MOMENTUM UPDATE

        self.vW = self.beta * self.vW + (1 - self.beta) * dW

        self.vb = self.beta * self.vb + (1 - self.beta) * db

        self.W -= learning_rate * self.vW

        self.b -= learning_rate * self.vb

        return dX

def softmax(Z):

    # Subtracting max(Z) prevents "Overflow" (Numerical Stability)

    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))

    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def plot_learning():

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)

        plt.plot(loss_history, label='Loss', color='red')

        plt.title('Loss Over Time')

        plt.xlabel('Epochs')

        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)

        plt.plot(accuracy_history, label='Accuracy', color='blue')

        plt.title('Accuracy Over Time')

        plt.xlabel('Epochs')

        plt.ylabel('Accuracy (%)')

        plt.show()

loss_history = []

accuracy_history = []

class Sequential:

    def __init__(self, layers):

        self.layers = layers

  

    def forward(self, X):

        for layer in self.layers:

            X = layer.forward(X)

        return X

  

    def backward(self, grad, lr):

        # The Palindrome: Moving backward through the list

        for layer in reversed(self.layers):

            # Check if layer has a learning rate (like Conv/Dense)

            if 'learning_rate' in layer.backward.__code__.co_varnames:

                grad = layer.backward(grad, lr)

            else:

                grad = layer.backward(grad)

        return grad

  

model = Sequential([

    # LAYER 1: The Edge Detector

    # Input: (Batch, 28, 28, 1) -> Output: (Batch, 28, 28, 8)

    ConvolutionLeyrer(in_channels=1, num_kernels=8, kernel_size=3),

    MaxPoolingLayer(), # Output: (Batch, 14, 14, 8)

  

    # LAYER 2: The Shape Detector

    # Input: (Batch, 14, 14, 8) -> Output: (Batch, 14, 14, 16)

    ConvolutionLeyrer(in_channels=8, num_kernels=16, kernel_size=3),

    MaxPoolingLayer(), # Output: (Batch, 7, 7, 16)

  

    FlattenLayer(),    # Bridge: 7 * 7 * 16 = 784

    # Updated input_size for Dense Layer (784)

    DenseLayer(input_size=784, output_size=10)

])

  

def train_step_v2(X_batch, Y_labels, lr=0.01):

    # 1. Forward

    probs = softmax(model.forward(X_batch))

    # 2. Loss (Judge)

    error = probs - Y_labels

    # 3. Backward (The automated chain)

    model.backward(error, lr)

    return np.mean(np.square(error))

  

path = r'E:\python\DATA ARCHITECTURE\MNIST\mnist_train.csv.csv'

print("Loading MNIST with NumPy... (Stay Patient, NumPy is a bit slower at I/O)")

data = np.loadtxt(path, delimiter=',', skiprows=1)

  

Y_raw = data[:, 0].astype(int)

X_raw = data[:, 1:]

X_train = X_raw.reshape(-1, 28, 28, 1) / 255.0

m = X_train.shape[0] # Total number of samples (60,000)

  

Y_train = np.zeros((m, 10))

Y_train[np.arange(m), Y_raw] = 1

  

print(f"Data Loaded Successfully. Shape: {X_train.shape}")

epochs = 6

batch_size = 24 # Small enough for your RAM, large enough for stable math

learning_rate = 0.01

  

for epoch in range(epochs):

    # 1. SHUFFLE: Mix the 60,000 images so the brain doesn't get bored

    permutation = np.random.permutation(m)

    X_shuffled = X_train[permutation]

    Y_shuffled = Y_train[permutation]

    total_loss = 0

    correct_preds = 0

    # 2. MINI-BATCH LOOP

    for i in range(0, m, batch_size):

        # Slice the data

        X_batch = X_shuffled[i : i + batch_size]

        Y_batch = Y_shuffled[i : i + batch_size]

        # --- FORWARD ---

        logits = model.forward(X_batch)

        probs = softmax(logits)

        # --- CALCULATE ACCURACY FOR THIS BATCH ---

        predictions = np.argmax(probs, axis=1)

        targets = np.argmax(Y_batch, axis=1)

        correct_preds += np.sum(predictions == targets)

        # --- BACKWARD ---

        error = (probs - Y_batch) # The "Shout" of the judge

        loss = np.mean(np.square(error))

        total_loss += loss

        model.backward(error, learning_rate)

        # 3. PRINT PROGRESS (Every 100 batches)

        if i % (batch_size * 1000) == 0:

            current_acc = (np.sum(predictions == targets) / batch_size) * 100

            print(f"Epoch {epoch} | Batch {i//batch_size} | Batch Acc: {current_acc:.2f}% | Loss: {loss:.6f}")

  
  

    # 4. END OF EPOCH SUMMARY

    avg_loss = total_loss / (m // batch_size)

    total_acc = (correct_preds / m) * 100

    print(f"\n=== EPOCH {epoch} FINISHED ===")

    print(f"Average Loss: {avg_loss:.6f} | Total Accuracy: {total_acc:.2f}%\n")

    loss_history.append(avg_loss)

    accuracy_history.append(total_acc)

plot_learning()