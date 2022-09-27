import json
import numpy as np


def getT(test_name):
    with open(test_name) as j:
        T = json.load(j)
    return T


def getPoses(T):

    P = np.array(T["poses"])
    P = np.array([[xy[:2] for xy in frame] for frame in P])

    # position moyenne de la nuque

    offx, offy = np.mean(P[:, 0], axis=0)

    # normalisation

    for i in range(len(P)):
        for j in range(len(P[i])):
            P[i][j][0] -= offx
            P[i][j][1] -= offy

    # concaténation par pack de frames

    return P


def getPosesRaw(T):

    P = np.array(T["poses"])
    P = np.array([[xy[:2] for xy in frame] for frame in P])

    return P
