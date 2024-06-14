'''
to realize the some statics functions, such as p-value, z-value and some acc parameters

'''

import os

dir_path = os.path.dirname(os.path.realpath(__file__))      # this path
# evaluate -> wmfunc
parent_dir_path = os.path.abspath(os.path.join(dir_path,    
                                               os.pardir)) 

import random
import sys
sys.path.append(parent_dir_path) # add path in system
from sklearn.metrics import confusion_matrix,roc_curve, auc
import matplotlib.pyplot as plt
from math import sqrt
import scipy
import numpy as np

def cal_z_score(observed_count, T,
                expected = 0.5):
    """ calculate z-score

    Args:
        observed_count (_type_): hit nums
        T (_type_): all nums
        expected (float, optional): expeted mean hit nums proportion. Defaults to 0.5.

    Returns:
        float: z-score
    """    
    numer = observed_count - expected * T
    denom = sqrt(T * expected * (1 - expected))
    z = numer / denom
    return z

def cal_p_value(z=None,
                use_z_score=True,
                observed_count = None, 
                T = None,
                expected = 0.5):
    """ calculate p-value

    Args:
        z (flaot, optional): z-score. Defaults to None.
        use_z_score (bool, optional): if use z-score to calculate p-value. Defaults to True.
        observed_count (_type_): hit nums
        T (_type_): all nums
        expected (float, optional): expeted mean hit nums proportion. Defaults to 0.5.

    Returns:
        float: p-value
    """    
    if not use_z_score:
        numer = observed_count - expected * T
        denom = sqrt(T * expected * (1 - expected))
        z = numer / denom
    p_value = scipy.stats.norm.sf(z)
    return p_value


def label_acc(true_labels, pred_labels):
    """calculate binary classifier

    Args:
        true_labels (_type_): true labels, larger num will be seemed to be positive samples
        pred_labels (_type_): pred labels, binary number as true labels

    Returns:
        _type_: precision, recall, accuracy, f1score
    """    
    # cal confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    
    # cal recall Sensitivity
    tpr = tp / (tp + fn)
    # cal fpr
    fpr = fp / (fp + tn)
    # cal Specificity
    tnr = tn / (fp + tn)
    # cal precision
    pre = tp / (tp + fp)
    # cal ACC
    acc = (tp + tn) / (tn + fp + fn + tp)
    # cal F1
    f1 = 2 * pre * tpr / (pre + tpr)
    
    return  pre, tpr, acc, f1, fpr

def logits_acc(true_labels, pred_logits, thresh = None, fpr = None):
    """cal by logic

    Args:
        true_labels (_type_): true labels
        pred_logits (_type_): predict logits, use threshold to get different fpr...
        fpr (_type_, optional): give fpr, to get target tpr and thresh. Defaults to None.
        thresh (_type_, optional): give thresh, get precision, recall, accuracy, f1score. Defaults to None.

    Returns:
        _type_: _description_
    """
    if thresh == None:
        # cal roc
        fprs, tprs, thresholds = roc_curve(true_labels, pred_logits, drop_intermediate=False)
        roc_auc = auc(fprs, tprs)
        
        if fpr == None:
            return fprs, tprs, roc_auc, thresholds
        else:
            idx = np.argmin(np.abs(fprs - fpr))
            
            fpr = fprs[idx]
            tpr = tprs[idx]
            thresh = thresholds[idx]
            return fpr, tpr, thresh
    else:
        # fix true labels
        assert set(true_labels) != 2, "true labels must be binary"
        f_label = min(set(true_labels))
        true_labels = [0 if label == f_label else 1 for label in true_labels]
        pred_labels = [0 if logits <= thresh else 1 for logits in pred_logits]
        pre, tpr, acc, f1, fpr = label_acc(true_labels, pred_labels)
        return pre, tpr, acc, f1, fpr

def calculate_roc_auc_and_performance(z_score_text, z_score_wm):
    """ to src text and watermark text to cal the acc...

    Args:
        z_score_text (List[float]): the src text detect score
        z_score_wm (List[float]): the wm text detect score
    """    
    record_dict = {}
    # get labels and scores
    labels = [0]*len(z_score_text) + [1]*len(z_score_wm)
    scores = z_score_text + z_score_wm

    # shuffle ?
    n = len(labels)
    shuffle_ids = random.sample(range(n), n)
    
    labels = [labels[i] for i in shuffle_ids]
    scores = [scores[i] for i in shuffle_ids]
    
    # cal fpr when tpr = 0.1/0.01
    fpr, tpr, thresh = logits_acc(labels, scores, fpr=0.1)
    record_dict["0.1"] = (fpr, tpr, thresh)
    fpr, tpr, thresh = logits_acc(labels, scores, fpr=0.01)
    record_dict["0.01"] = (fpr, tpr, thresh)

    fprs, tprs, roc_auc, thresh = logits_acc(labels, scores)

    record_dict["area"] = roc_auc
    
    return record_dict, fprs, tprs, thresh


def run_demo():
    n = 500
    # 生成长度为n的浮点数列表，范围在0到1之间
    float_list = np.random.rand(n)

    # 生成长度为n的0或1的整数列表
    int_list = np.random.randint(2, size=n).astype(int)
    a,b,c,d = logits_acc(pred_logits=float_list, true_labels=int_list, thresh=0.1)
    print(f"fpr: {a}  tpr:{b}   thresh:{c}")
    
# TODEL
# run_demo()
