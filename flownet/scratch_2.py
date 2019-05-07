import numpy as np
import matplotlib.pyplot as plt

def calc_error(flow_field_1, flow_field_2):
    """ Docstring """
    return np.mean(np.linalg.norm(flow_field_2 - flow_field_1, axis=1))


def compare_field_sequences(seq_1, seq_2):
    """ Docstring """
    n1, h1, w1, c1 = seq_1.shape
    n2, h2, w2, c2 = seq_2.shape
    assert h1 == h2, "Height of input sequences do not match."
    assert w1 == w2, "Width of input sequences do not match."
    assert c1 == c2, "Depth of input sequences do not match."

    # guarantee that seq 1 is longer or equal in length to seq 2
    if n1 < n2:
        return compare_field_sequences(
            seq_1=seq_2,
            seq_2=seq_1)

    errors_by_overlap = []
    # slide the shorter sequence over the longer
    for move in range(n1 + 2):
        # window start idx
        k = move - n2 + 1

        if k < 0 or k > n1-n2:
            seq_1_temp = np.zeros((n2, h1, w1, c1), dtype=np.float32)
            for x in range(max(k, 0), min(n1, move+1)):
                seq_1_temp[x-k] = seq_1[x]
            errors_by_overlap.append(calc_error(seq_1_temp, seq_2))
        else:
            errors_by_overlap.append(calc_error(seq_1[k:k+n2], seq_2))

    return errors_by_overlap


s1 = np.ones((8, 2, 2, 2), dtype=np.float32)
s2 = np.ones((3, 2, 2, 2), dtype=np.float32)

# print(s1)
# print(s2)

errors_by_overlap = compare_field_sequences(s1, s2)
print(errors_by_overlap)

plt.plot(range(len(errors_by_overlap)), errors_by_overlap)
plt.show()