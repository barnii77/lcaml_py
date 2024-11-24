def sin(x, N_TERMS):
    pi = 3.14159265358979
    x = x % (2 * pi)
    sum_term = 0.0
    n = 0
    sign = 0
    m = 0
    fact = 0.0
    pow = 0.0

    while n < N_TERMS:
        #  sign
        if n % 2 == 0:
            sign = 1
        else:
            sign = -1

        #  power
        m = 2 * n + 1
        pow = 1.0
        while m > 0:
            pow = pow * x
            m = m - 1

        #  factorial
        m = 2 * n + 1
        fact = 1.0
        while m > 1:
            fact = fact * m
            m = m - 1

        #  sine  term
        sum_term = sum_term + sign * pow / fact
        n = n + 1
    return sum_term


pi = 3.14159265358979
N = 4000
sin_pi = 0.0

sin_out = sin(pi, N)

if abs(sin_out - sin_pi) > 0.0001:
    raise Exception("incorrect  output")
