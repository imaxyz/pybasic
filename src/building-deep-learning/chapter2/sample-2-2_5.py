from sample2_3_3 import AND, NAND, OR


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)

    return y


def main():
    print('---XOR---')
    print(f'(0, 0): {XOR(0, 0)}')
    print(f'(1, 0): {XOR(1, 0)}')
    print(f'(0, 1): {XOR(0, 1)}')
    print(f'(1, 1): {XOR(1, 1)}')


if __name__ == '__main__':
    main()
