from matplotlib import pyplot as plt
from scipy.fftpack import fft
from random import randint
import numpy as np
import time
import algs
import scipy.stats as sps
import scipy.special as ssp
import math
import trng

def SqueezeTest(__RandomSequence):
    arr = [0 for i in range(50)]
    #rd = MakeMidSqSequence(100000*20)
    rd = __RandomSequence(100000*20, [0, 10000])
    print(rd[0:100])
    for i in range(100000):
        n = 2**31
        j = 0
        while n >= 1:
            #n = n * (randint(0, 10000) / 10000)
            n = n * ((rd[i+j] % 10001) / 10000)
            j += 1
        arr[j] += 1

    plt.bar([i for i in range(50)], arr)
    plt.show()

def MonteCarloTest(__RandomSequence):
    k = 0
    #plt.ion()
    plt.plot(np.linspace(0, np.pi, 100), np.sin(np.linspace(0, np.pi, 100)))
    #rd = algs.MakeMidSqSequence(100000*20)
    rd = __RandomSequence(100000, [0, 10000])
    for i in range(1000):
        plt.title(f"n = {i+1},\n value = {(k/(i+1) * np.pi):.4f}")
        #x = randint(0, 10000) / 10000 * np.pi
        #y = randint(0, 10000) / 10000
        x = rd[i] / 10000 * np.pi
        y = rd[i+10001] / 10000
        if y <= np.sin(x):
            plt.plot([x], [y], 'go')
            k += 1
        else:
            plt.plot([x], [y], 'bo')
        plt.pause(0.0001)
    
    
    plt.show()

""" def BitTest(__RandomSequence):
    rd = __RandomSequence(10000000, [0, 1])
    s = ''.join([str(i) for i in rd])
    print(abs(s.count('0')/len(s) - 0.5)) """

def ExpectedValue(__RandomSequence):
    rd = __RandomSequence(100000, [0, 10000])
    rd = np.array(rd)
    print(rd.mean())
    print(rd.var())

def FourierTest(__RandomSequence, n = 1000):
    rd = __RandomSequence(n, [0, 1])
    rd = np.array([-1 if i == 0 else 1 for i in rd])
    spectral = fft(rd)
    slc = math.floor(n/2)
    mod = abs(spectral[0:slc])
    tau = np.sqrt(np.log(1 / 0.05) * n )
    n0 = 0.95 * (n / 2)
    n1 = len(np.where(mod < tau)[0])
    d = (n1 - n0) / np.sqrt(n * (0.95) * (0.05) / 4)
    p_value = math.erfc(math.fabs(d) / np.sqrt(2))
    print(p_value)
    #plt.plot(mod)
    #plt.show()

def BitTest(__RandomSequence, n = 1000):
    rd = __RandomSequence(n, [0, 1])
    s = ''.join([str(i) for i in rd])
    sm = 0
    for i in s:
        if i == '0':
            sm -= 1
        else:
            sm += 1
    
    print(math.erfc((abs(sm) / np.sqrt(n))/np.sqrt(2)))

def BlockTest(__RandomSequence, n = 1000):
    block = 64
    rd = __RandomSequence(n, [0, 1])
    block_start = 0
    block_end = block
    proportion_sum = 0.0

    for counter in range(n // block):
        block_data = rd[block_start:block_end]

        one_count = 0
        for bit in block_data:
            if bit == 1:
                one_count += 1
        pi = one_count / block

        proportion_sum += pow(pi - 0.5, 2.0)

        block_start += block
        block_end += block

    result = 4.0 * block * proportion_sum

    p_value = ssp.gammaincc((n // block) / 2, result / 2)

    print(p_value)

def RunTest(__RandomSequence, n = 1000):
    rd = __RandomSequence(n, [0, 1])
    one_count = 0
    vObs = 0
    length_of_binary_data = len(rd)

    tau = 2 / np.sqrt(length_of_binary_data)

    one_count = rd.count(1)

    pi = one_count / length_of_binary_data

    if abs(pi - 0.5) >= tau:
        print(False)
        return
    else:
        for item in range(1, length_of_binary_data):
            if rd[item] != rd[item - 1]:
                vObs += 1
        vObs += 1
        p_value = math.erfc(abs(vObs - (2 * (length_of_binary_data) * pi * (1 - pi))) / (2 * math.sqrt(2 * length_of_binary_data) * pi * (1 - pi)))
    
    print(p_value)

def LongestRunOfOnesInBlockTest(__RandomSequence, n = 1000):
    rd = __RandomSequence(n, [0, 1])
    one_count = 0
    length_of_binary_data = len(rd)
    if length_of_binary_data < 128:
        # Not enough data to run this test
        print(False)
        return
    elif length_of_binary_data < 6272:
        k = 3
        m = 8
        v_values = [1, 2, 3, 4]
        pi_values = [0.21484375, 0.3671875, 0.23046875, 0.1875]
    elif length_of_binary_data < 750000:
        k = 5
        m = 128
        v_values = [4, 5, 6, 7, 8, 9]
        pi_values = [0.1174035788, 0.242955959, 0.249363483, 0.17517706, 0.102701071, 0.112398847]
    else:
        # If length_of_bit_string > 750000
        k = 6
        m = 10000
        v_values = [10, 11, 12, 13, 14, 15, 16]
        pi_values = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

    number_of_blocks = math.floor(length_of_binary_data / m)
    block_start = 0
    block_end = m
    xObs = 0
    # This will intialized an array with a number of 0 you specified.
    frequencies = np.zeros(k + 1)

    # print('Number of Blocks: ', number_of_blocks)

    for count in range(number_of_blocks):
        block_data = rd[block_start:block_end]
        max_run_count = 0
        run_count = 0

        # This will count the number of ones in the block
        for bit in block_data:
            if bit == 1:
                run_count += 1
                max_run_count = max(max_run_count, run_count)
            else:
                max_run_count = max(max_run_count, run_count)
                run_count = 0

        max(max_run_count, run_count)

        #print('Block Data: ', block_data, '. Run Count: ', max_run_count)

        if max_run_count < v_values[0]:
            frequencies[0] += 1
        for j in range(k):
            if max_run_count == v_values[j]:
                frequencies[j] += 1
        if max_run_count > v_values[k - 1]:
            frequencies[k] += 1

        block_start += m
        block_end += m

    # print("Frequencies: ", frequencies)
    # Compute xObs
    for count in range(len(frequencies)):
        xObs += pow((frequencies[count] - (number_of_blocks * pi_values[count])), 2.0) / (
                number_of_blocks * pi_values[count])

    p_value = ssp.gammaincc(float(k / 2), float(xObs / 2))

    print(p_value)

def ApproximateEntropyTest(__RandomSequence, n = 1000):
    pattern_length = 10
    rd = __RandomSequence(n, [0, 1])
    binary_data = ''.join([str(i) for i in rd])
    length_of_binary_data = len(binary_data)

    binary_data += binary_data[:pattern_length + 1:]

    max_pattern = ''
    for i in range(pattern_length + 2):
        max_pattern += '1'

    vobs_01 = np.zeros(int(max_pattern[0:pattern_length:], 2) + 1)
    vobs_02 = np.zeros(int(max_pattern[0:pattern_length + 1:], 2) + 1)

    for i in range(length_of_binary_data):
        vobs_01[int(binary_data[i:i + pattern_length:], 2)] += 1
        vobs_02[int(binary_data[i:i + pattern_length + 1:], 2)] += 1

    vobs = [vobs_01, vobs_02]

    sums = np.zeros(2)
    for i in range(2):
        for j in range(len(vobs[i])):
            if vobs[i][j] > 0:
                sums[i] += vobs[i][j] * np.log(vobs[i][j] / length_of_binary_data)
    sums /= length_of_binary_data
    ape = sums[0] - sums[1]

    xObs = 2.0 * length_of_binary_data * (np.log(2) - ape)

    p_value = ssp.gammaincc(pow(2, pattern_length - 1), xObs / 2.0)

    print(p_value)

def CumulativeSumsTest(__RandomSequence, n = 1000):
    rd = __RandomSequence(n, [0, 1])
    binary_data = ''.join([str(i) for i in rd])
    n = len(binary_data)
    counts = np.zeros(n)
    mode = 0
    if not mode == 0:
        binary_data = binary_data[::-1]

    counter = 0
    for char in binary_data:
        sub = 0
        if char == '0':
            sub = -1
        if char == '1':
            sub = 1
        if counter > 0:
            counts[counter] = counts[counter -1] + sub
        else:
            counts[counter] = sub

        counter += 1
    #print(counts)
    z = max(abs(counts))
    #print(z)
    start = int(np.round(0.25 * (-n / z + 1)))
    end = int(np.round(0.25 * (n / z - 1)))

    terms_one = []
    for k in range(start, end + 1):
        sub = sps.norm.cdf((4 * k - 1) * z / np.sqrt(n))
        terms_one.append(sps.norm.cdf((4 * k + 1) * z / np.sqrt(n)) - sub)

    start = int(np.floor(0.25 * (-n / z - 3)))
    end = int(np.floor(0.25 * (n / z - 1)))

    terms_two = []
    for k in range(start, end + 1):
        sub = sps.norm.cdf((4 * k + 1) * z / np.sqrt(n))
        terms_two.append(sps.norm.cdf((4 * k + 3) * z / np.sqrt(n)) - sub)

    p_value = 1.0 - sum(terms_one)
    p_value += sum(terms_two)

    print(p_value)


size = 1000000
'''
print('Monobit test: ', end=' ')
BitTest(algs.LCGsequence, size)
print('Freq test within a block: ', end=' ')
BlockTest(algs.LCGsequence, size)
print('Runs test: ', end=' ')
RunTest(algs.LCGsequence, size)
print('Test for the longest run of ones in a block: ', end=' ')
LongestRunOfOnesInBlockTest(algs.LCGsequence, size)
print('Spectral test: ', end=' ')
FourierTest(algs.LCGsequence, size)
print('Approximate entropy test: ', end=' ')
ApproximateEntropyTest(algs.LCGsequence, size)
'''
print('Cumulative sums test', end=' ')
CumulativeSumsTest(algs.LCGsequence, size)