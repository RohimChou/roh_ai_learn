import time
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, learning_rate=0.05):
        # Initialize with random weights and bias
        self.weights = np.random.rand(1)  # Just one input
        self.bias = np.random.rand(1)  # Bias term
        self.learning_rate = learning_rate

    def activate(self, x):
        # Identity activation: output = input (linear). Makes the whole net a linear function.
        return x

    def activate_fun_derivative(self, x):
        # Derivative of identity activation is 1 (used in backprop)
        return 1

    def predict(self, x):
        # Forward pass: calculate the weighted sum, then apply activation
        # Need to reshape x for matrix multiplication
        x_input = np.array([x])
        sum = np.dot(x_input, self.weights) + self.bias
        predicted_value = self.activate(sum)
        return predicted_value

    def train(self, training_inputs, training_outputs, epochs=200):
        # Normalize outputs to [0,1] for activation
        max_output = np.max(training_outputs)
        min_output = np.min(training_outputs)
        # normalized_outputs = (training_outputs - min_output) / (max_output - min_output)

        for epoch in range(epochs):
            total_error = 0
            # For each training example
            for i, (x, expected_y) in enumerate(zip(training_inputs, training_outputs)):
                # Step 1: Predict (forward pass)
                prediction = self.predict(x)

                # Step 2: Calculate error
                error = expected_y - prediction
                total_error += abs(error[0])

                # Step 3: Adjust weights and bias (backpropagation)
                adjustment = error * self.activate_fun_derivative(prediction)
                self.weights += self.learning_rate * adjustment * x
                self.bias += self.learning_rate * adjustment

                print(f"\rEpoch {epoch + 1:>4}, "
                      f"Error: {total_error / len(training_inputs):>6.5f}, "
                      f"Weight: {self.weights[0]:>6.4f}, "
                      f"Bias: {self.bias[0]:>6.4f}", end='', flush=True)
                time.sleep(0.05)

                # Show progress only on last item of each epoch
                # if i == len(training_inputs) - 1:
                #     print(f"\rEpoch {epoch + 1:>4}, "
                #           f"Error: {total_error / len(training_inputs):>6.5f}, "
                #           f"Weight: {self.weights[0]:>6.4f}, "
                #           f"Bias: {self.bias[0]:>6.4f}", end='', flush=True)
                #     time.sleep(0.05)

        print("\nTraining complete.")
        # Store normalization parameters for predictions later
        self.min_output = min_output
        self.max_output = max_output