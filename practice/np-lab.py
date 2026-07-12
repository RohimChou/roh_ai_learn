import numpy as np

if __name__ == "__main__":
    print("Hello, World!")
    print(np.absolute(-5))

    print("\nnp.array([1, 2, 3]) + np.array([4, 5, 6])")
    print(np.array([1, 2, 3]) + np.array([4, 5, 6]))

    # 2x3 * 3x2 = 2x2
    a = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    print("\nnp.dot(2x3, 3x2)")
    print(np.dot(a, b))

    # random
    print("\nnp.random.rand(2, 3)")
    print(np.random.rand(2, 3))