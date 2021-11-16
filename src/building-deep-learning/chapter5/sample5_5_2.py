from layers import *

def main():

    sigmoid_layer = SigmoidLayer()

    x = np.array([1, -125, 2, -3.5134, 0.01])
    result = sigmoid_layer.forward(x)

    result2 = sigmoid_layer.backward(1)

    pass

if __name__ == '__main__':
    main()


