import numpy as np

from model import HMM
from data import *


class Viterbi(HMM):
    def __init__(self, A: np.array, B: np.array, pi: np.array):
        super().__init__(A, B, pi)

    def forward(self, observations):

        N, M, T = self.paramters()

        alpha = np.zeros((T, N))
        back_pointer = np.zeros((T, N))
        # Initialization
        alpha[0, :] = self.pi * self.B[:, observations[0]]
        back_pointer[0, :] = np.array([0, 1])
        # Recursion
        for t in range(1, T):
            o = observations[t]
            for s in range(N):
                b = self.B[s, o]
                alpha[t, s] = np.max(alpha[t - 1, :] * (b * self.A[:, s]))
                back_pointer[t, s] = np.argmax(alpha[t - 1, :] * (b * self.A[:, s]))

        alpha_sum = np.sum(alpha, axis=0)
        best_score = np.max(alpha_sum)
        best_path = back_pointer[:, np.argmax(alpha_sum)]
        return alpha, back_pointer, best_score, best_path


if __name__ == "__main__":
    hmm = Viterbi(A, B, pi)
    alpha, back_pointer, best_score, best_path = hmm.forward(observations)
