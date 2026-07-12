import numpy as np
from simple_neural_network import SimpleNeuralNetwork

if __name__ == "__main__":
    # AND logic gate: both inputs must be 1 for output to be 1
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 0, 0, 1])

    # Create and train the neural network
    # 拓撲：單一顆 neuron，無隱藏層，可學參數只有 3 個（w₁、w₂、b）
    #
    #  x₁ ●──── w₁ ─────┐
    #                   ├───▶ Σ = x₁w₁ + x₂w₂ + b ──▶ sigmoid ──▶ ŷ (0~1)
    #  x₂ ●──── w₂ ─────┘                       ▲
    #               bias b ─────────────────────┘
    #
    # 幾何意義：學一條直線 x₁w₁ + x₂w₂ + b = 0 把平面切兩半，
    # (1,1) 在一側（→1）、其他三點在另一側（→0）。
    # AND 線性可分所以單 neuron 就夠；XOR 切不開，得加隱藏層。
    nn = SimpleNeuralNetwork()
    nn.train(training_inputs, training_outputs)

    # Test the trained network
    print("\nTesting the trained neural network on AND logic:")
    for inputs in training_inputs:
        prediction = nn.predict(inputs)
        print(f"Inputs: {inputs}, Prediction: {prediction:.4f}, Rounded: {round(prediction)}")