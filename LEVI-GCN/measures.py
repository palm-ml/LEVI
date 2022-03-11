import numpy as np 
import math


def chebyshev(rd,pd):
    temp = np.abs(rd-pd)
    temp = np.max(temp,1)
    distance = np.mean(temp)
    return distance

def clark(rd,pd):
    temp1 = (pd - rd)**2
    temp2 = (pd + rd)**2
    temp = np.sqrt(np.sum(temp1 / temp2, 1))
    distance = np.mean(temp)
    return distance

def canberra(rd,pd):
    temp1 = np.abs(rd-pd)
    temp2 = rd + pd
    temp = np.sum(temp1 / temp2,1)
    distance = np.mean(temp)
    return distance

def kl_dist(rd,pd):
    eps = 1e-12
    temp = rd * np.log(rd / pd + eps)
    temp = np.sum(temp,1)
    distance = np.mean(temp)
    return distance

def cosine(rd,pd):
    inner = np.sum(pd*rd,1)
    temp1 = np.sqrt(np.sum(pd**2,1))
    temp2 = np.sqrt(np.sum(rd**2,1))
    temp = inner / (temp1*temp2)
    distance = np.mean(temp)
    return distance


def intersection(rd,pd):
    (rows,cols) = np.shape(rd)
    dist = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            dist[i] = dist[i] + min(rd[i,j],pd[i,j])
    distance = np.mean(dist)
    return distance

if __name__ == '__main__':
    rd = np.array([[0.24,0.17,0.16],[0.37,0.12,0.20]])
    pd = np.array([[0.23,0.22,0.21],[0.25,0.12,0.25]])
    # dist1 = chebyshev(rd,pd)
    # dist2 = clark(rd,pd)
    # dist3 = kl_dist(rd,pd)
    # dist4 = cosine(rd,pd)
    dist5 = intersection(rd,pd)
    print(dist5)