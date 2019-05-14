import numpy as np

#score function
def f(x):
    y=np.abs(x-5.5)
    res=np.exp(y)-1;
    return res

# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    nsi=[f(i) for i in si]
    #print(nsi)
    nsi=np.array(nsi)
    nsi=nsi/nsi.sum()
    vec=scores*nsi;
    vec=vec/vec.sum();
    mean = np.sum(vec*si)
    return mean

# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std
