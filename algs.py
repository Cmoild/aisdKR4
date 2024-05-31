import time

def MiddleSquare(seed_number, interval):
    number = seed_number
    already_seen = set()
    counter = 1
    ret = []
    lower, upper = interval[0], interval[1]

    while number not in already_seen:
        counter += 1
        already_seen.add(number)
        number = int(str(number * number).zfill(20)[5:15])
        ret.append(round(number % (upper - lower + 1) + lower))
    return ret

def MidSqSequence(n, interval, seed=0):
    ret = []
    i = str(time.time()).replace(".", "")
    i = int(i[len(i)-10:len(i)])
    i = 1000000001
    while len(ret) < n:
        ret += MiddleSquare(i, interval)
        i += 1
    #print(len(ret))
    return ret[0:n]

def lcg(x, a, c, m):
    while True:
        x = (a * x + c) % m
        yield x


def LCGsequence(n, interval, seed=0):
    a, c, m = 1103515245, 12345, 2 ** 31
    func = lcg(seed, a, c, m)

    lower, upper = interval[0], interval[1]
    ret = []
    for i in range(n):
        observation = (upper - lower) * (next(func) / (2 ** 31 - 1)) + lower
        ret.append(round(observation))

    return ret

#print(random_uniform_sample(10, [0, 1000]))
def xorshift(x):
    while True:
        x = x ^ (x << 13)
        x = x ^ (x >> 17)
        x = x ^ (x << 5)
        x = x % (2 ** 32 - 1)
        yield x

def XorshiftSequence(n, interval, seed=1):
    ret = []
    func = xorshift(seed)

    lower, upper = interval[0], interval[1]

    for i in range(n):
        observation = (upper - lower) * (next(func) / (2 ** 32 - 1)) + lower
        ret.append(round(observation))

    return ret


#print(LCGsequence(10, [0, 1]))