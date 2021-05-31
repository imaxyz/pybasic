a = int(input())
# print(a)

b_c = list(map(int, input().split(sep=' ')))
# print(b_c)

s = input()
# print(s)

abc_sum = a + sum(b_c)
output = f'{abc_sum} {s}'
print(output)
