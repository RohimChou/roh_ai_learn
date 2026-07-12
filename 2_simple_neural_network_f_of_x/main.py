import numpy as np
from simple_neural_network import SimpleNeuralNetwork

if __name__ == "__main__":
    # f(x) = 2x + 3
    training_inputs = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # training_outputs = np.array([5, 7, 9, 11, 13, 15, 17, 19])
    training_outputs = training_inputs * 5 + 11

    # Create and train the neural network
    nn = SimpleNeuralNetwork()
    nn.train(training_inputs, training_outputs)

    # Test the trained network
    print("\nTesting the trained neural network on f(x) = 5x + 11:")
    for inputs in np.array([1, 2, 3, 10, 20]):
        prediction = nn.predict(inputs)
        # prediction_denormalized = (prediction * (np.max(training_outputs) - np.min(training_outputs))) + np.min(training_outputs)

        print(f"Inputs: {inputs}, Expected: {5 * inputs + 11}, Prediction: {prediction[0]:.4f}")