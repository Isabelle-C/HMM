import numpy as np

from model import HMM
from data import *


class BaumWelch(HMM):
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
        return alpha

    def backward(self, observations):
        N, M, T = self.paramters()

        # Initialization
        beta = np.zeros((T, N))
        beta[-1, :] = 1
        # Recursion
        for t in range(T - 2, -1, -1):
            tplus1_o = observations[t + 1]
            for s in range(N):
                b = self.B[s, tplus1_o]
                beta[t, s] = np.sum(beta[t + 1, :] * (b * self.A[:, s]))
        # print(np.sum(beta[0]))
        return beta

    def calculate_xi_gamma(self, alpha, beta, observations):
        N, M, T = self.paramters()
        xi = np.zeros((T - 1, N, N))
        gamma = np.zeros((T, N))

        for i in range(T - 1):
            tplus1_o = observations[i + 1]
            for j in range(N):
                xi_num = (
                    alpha[i, j] * self.A[j, :] * self.B[:, tplus1_o] * beta[i + 1, :]
                )
                xi_den = np.sum(alpha[i, :] * beta[i, :])
                xi[i, j, :] = xi_num / xi_den

                gamma[i, j] = np.sum(xi[i, :, j])

        gamma[T - 1, :] = (
            alpha[T - 1, :]
            * beta[T - 1, :]
            / (np.sum(alpha[T - 1, :] * beta[T - 1, :]))
        )
        return xi, gamma

    def run(self, observations, max_iter=10, tol=1e-4):

        N, M, T = self.paramters()

        for iteration in range(max_iter):
            # E-step

            alpha = self.forward(observations)
            beta = self.backward(observations)

            xi, gamma = self.calculate_xi_gamma(alpha, beta, observations)

            # M-step
            new_A = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    epsilon = 1e-10  # Small value to avoid division by zero

                    new_A_num = np.sum(xi[:, i, j])
                    new_A_den = (
                        np.sum(gamma[:-1, i]) + epsilon
                    )  # Add epsilon to avoid division by zero
                    new_A[i, j] = np.clip(
                        new_A_num / new_A_den, 0, 1
                    )  # Clip the value to avoid overflow

            new_B = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    new_B_num = np.sum(
                        [gamma[t, i] for t in range(T) if observations[t] == j]
                    )  # only count the observation with j
                    new_B_den = np.sum(gamma[:, i])
                    new_B[i, j] = new_B_num / new_B_den

            self.A = new_A
            self.B = new_B

            # Check for convergence
            if np.max(abs(new_A - A)) < tol and np.max(abs(new_B - B)) < tol:
                print(f"Converged after {iteration+1} iterations")
                break


if __name__ == "__main__":
    hmm = BaumWelch(A, B, pi)
    print("---Before---")
    print(A)
    print(B)
    hmm.run(observations)
    print("---After---")
    print(hmm.A)
    print(hmm.B)
