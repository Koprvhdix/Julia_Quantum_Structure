import numpy as np

def swap_chars(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


np.set_printoptions(threshold=np.inf)

N = 3

for num1 in range(N):
    for num2 in range(num1 + 1, N):
        the_matrix = np.zeros([2 ** N, 2 ** N])
        print(num1, num2)
        for number in range(2 ** N):
            number_str = format(number, '0{}b'.format(N))
            number_str = swap_chars(number_str, num1, num2)
            number_23 = int(number_str, 2)
            the_matrix[number, number_23] = 1
        print(the_matrix)