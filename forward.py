import numpy as np

from model import HMM
from data import *


class Forward(HMM):
    def __init__(self, A: np.array, B: np.array, pi: np.array):
        super().__init__(A, B, pi)

    def forward(self, observations):

        N, M, T = self.paramters()

        alpha = np.zeros((T, N))
        # Initialization
        alpha[0, :] = self.pi * self.B[:, observations[0]]
        # Recursion
        for t in range(1, T):
            for s in range(N):
                o = observations[t]
                b = self.B[s, o]
                alpha[t, s] = np.sum(alpha[t - 1, :] * (b * self.A[:, s]))

        print(alpha)
        return np.sum(alpha[-1])


if __name__ == "__main__":
    hmm = Forward(A, B, pi)
    forward_prob = hmm.forward(observations)
    print(forward_prob)
