from layers import *

def main():

    relu = ReLU()

    x = np.array([1, -125, 2, -3.5134, 0.01])
    result = relu.forward(x)

    pass

if __name__ == '__main__':
    main()


