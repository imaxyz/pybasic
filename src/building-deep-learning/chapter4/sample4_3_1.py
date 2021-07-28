def func(x):
    return x*x


def numerical_diff(f, x):
    h = 1e-4
    return (func(x=x+h) - func(x=x-h)) / (2*h)


def main():

    result = numerical_diff(func, 15)

    print('result: ', result)


if __name__ == '__main__':
    main()
