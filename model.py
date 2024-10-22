import numpy as np


class HMM:
    """
    Hidden Markov Model for Baum-Welch Algorithm

    Attributes
    ----------
    A : np.array
        Transition probability matrix.
    B : np.array
        Emission probability matrix.
    pi : np.array
        Initial probability distribution over states.
    """

    def __init__(self, A: np.array, B: np.array, pi: np.array):
        self.A: np.array = A
        self.B: np.array = B
        self.pi: np.array = pi

    def paramters(self):
        N = 2  # The number of states
        M = 3  # The number of observations
        T = 3  # The length of observation sequence

        return N, M, T
