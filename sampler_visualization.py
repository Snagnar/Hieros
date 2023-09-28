from embodied.replay.selectors import EfficientTimeBalanced, TimeBalancedNaive, Uniform
import numpy as np
from tqdm import tqdm

# num_steps = 1000
num_steps = 1000
num_large_steps = num_steps
# num_large_steps = 1000 * num_steps
# num_large_steps = 100 * num_steps
arrn = np.zeros((num_steps * 4,))
arrf = np.zeros(
    (
        5,
        num_large_steps * 4,
    )
)
arru = np.zeros((num_large_steps * 4,))

tn = TimeBalancedNaive()
# tf1 = EfficientTimeBalanced(length=1e6, temperature=1.0)
tf1 = EfficientTimeBalanced(length=num_large_steps * 4, temperature=0.99)
# tf1 = EfficientTimeBalanced(length=1e6, temperature=0.7)
tf2 = EfficientTimeBalanced(length=num_large_steps * 4, temperature=0.7)
tf3 = EfficientTimeBalanced(length=num_large_steps * 4, temperature=0.3)
tf4 = EfficientTimeBalanced(length=num_large_steps * 4, temperature=0.1)
tf5 = EfficientTimeBalanced(length=num_large_steps * 4, temperature=0.5)
un = Uniform()

for i in tqdm(range(num_large_steps)):
    for j in range(4):
        tf1[4 * i + j] = i + j
    for k in range(16):
        sf = tf1()
        arrf[0][sf] += 1
for i in tqdm(range(num_large_steps)):
    for j in range(4):
        tf2[4 * i + j] = i + j
    for k in range(16):
        sf = tf2()
        arrf[1][sf] += 1
for i in tqdm(range(num_large_steps)):
    for j in range(4):
        tf3[4 * i + j] = i + j
    for k in range(16):
        sf = tf3()
        arrf[2][sf] += 1
for i in tqdm(range(num_large_steps)):
    for j in range(4):
        tf4[4 * i + j] = i + j
    for k in range(16):
        sf = tf4()
        arrf[3][sf] += 1
for i in tqdm(range(num_large_steps)):
    for j in range(4):
        tf5[4 * i + j] = i + j
    for k in range(16):
        sf = tf5()
        arrf[4][sf] += 1


for i in tqdm(range(num_large_steps)):
    un[4 * i + 0] = i + 0
    un[4 * i + 1] = i + 1
    un[4 * i + 2] = i + 2
    un[4 * i + 3] = i + 3

    for _ in range(16):
        su = un()
        arru[su] += 1

for i in tqdm(range(num_steps)):
    tn[4 * i + 0] = i
    tn[4 * i + 1] = i
    tn[4 * i + 2] = i
    tn[4 * i + 3] = i

    for _ in range(16):
        sn = tn()
        arrn[sn] += 1

meann = np.mean(arrn)
meanf1 = np.mean(arrf[0])
meanf2 = np.mean(arrf[1])
meanf3 = np.mean(arrf[2])
meanf4 = np.mean(arrf[3])
meanf5 = np.mean(arrf[4])
meanu = np.mean(arru)

print("mean departure from uniform:", np.mean(np.abs(arrn - meann)))
print("mean departure from uniform fancy 1:", np.mean(np.abs(arrf[0] - meanf1)))
print("mean departure from uniform fancy 2:", np.mean(np.abs(arrf[1] - meanf2)))
print("mean departure from uniform fancy 3:", np.mean(np.abs(arrf[2] - meanf3)))
print("mean departure from uniform fancy 4:", np.mean(np.abs(arrf[3] - meanf4)))
print("mean departure from uniform fancy 5:", np.mean(np.abs(arrf[4] - meanf5)))
print("mean departure from uniform old:", np.mean(np.abs(arru - meanu)))
print("variance:", np.var(arrn))
print("variance fancy 1:", np.var(arrf[0]))
print("variance fancy 2:", np.var(arrf[1]))
print("variance fancy 3:", np.var(arrf[2]))
print("variance fancy 4:", np.var(arrf[3]))
print("variance fancy 5:", np.var(arrf[4]))
print("variance old:", np.var(np.abs(arru - meanu)))
print(np.var(arrn))
print(np.var(arrf))
print(np.var(arru))

print("zero percent naive:", np.sum(arrn == 0) / len(arrn))
print("zero percent fancy:", np.sum(arrf[0] == 0) / len(arrf[0]))


def smooth_values(arr, window=100):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = np.mean(arr[max(0, i - window) : i + window])
    return result


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))


def visualize_array(arr, suffix=""):
    arr = smooth_values(arr)
    plt.plot(arr, label=suffix)


# visualize_array(arrn, "TBS")
visualize_array(arrf[0], f"ETBS τ=1.0")
visualize_array(arrf[1], f"ETBS τ=0.7")
visualize_array(arrf[2], f"ETBS τ=0.3")
visualize_array(arru, "uniform")

plt.legend()
plt.savefig(f"combined.png")
