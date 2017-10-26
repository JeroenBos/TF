from math import sin, pi

INPUT_SIZE = 10
DOMAIN_MAX = 2 * pi

sin_input = [i * DOMAIN_MAX / INPUT_SIZE for i in range(INPUT_SIZE)]
sin_output = [sin(x) for x in sin_input]


for (a, b) in zip(sin_input, sin_output):
    print(a, b)
