import numpy as np


def BHS_standard(err):
    """
		Computes the BHS Standard metric
    """

    leq5 = 0
    leq10 = 0
    leq15 = 0

    for i in range(len(err)):

        if (abs(err[i]) <= 5):
            leq5 += 1
            leq10 += 1
            leq15 += 1

        elif (abs(err[i]) <= 10):
            leq10 += 1
            leq15 += 1

        elif (abs(err[i]) <= 15):
            leq15 += 1

    return (leq5 * 100.0 / len(err), leq10 * 100.0 / len(err), leq15 * 100.0 / len(err))


