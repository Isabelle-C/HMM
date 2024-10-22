import numpy as np

# Example 1 seen in the textbook
observations = [i - 1 for i in [3, 1, 3]]  # [,[1,1,2],[1,2,3]]
actual_states = [
    0 if s == "cold" else 1 for s in ["hot", "hot", "cold"]
]  # [,['cold','cold','cold'],['cold','hot','hot']]

pi = [0.2, 0.8]

A = np.array([[0.5, 0.5], [0.4, 0.6]])  # N by N matrix
B = np.array([[0.5, 0.4, 0.1], [0.2, 0.4, 0.4]])  # N by M matrix
pi = np.array([0.2, 0.8])

# -- Example 2
# # Transition matrix (A)
# A = np.array([
#     [0.7, 0.3],
#     [0.4, 0.6]
# ])

# # Emission matrix (B)
# B = np.array([
#     [0.1, 0.4, 0.5],
#     [0.7, 0.2, 0.1]
# ])

# # Initial state distribution (pi)
# pi = np.array([0.6, 0.4])

# # Observations (sequence of observed states)
# observations = np.array([0, 1, 2, 1, 0])
