import numpy as np

def bbox2slice(bbox):
    return np.index_exp[bbox[1]:bbox[3], bbox[0]:bbox[2]]