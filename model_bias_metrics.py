import numpy as np
import sklearn.metrics

## Auxiliary functions

def class_fpr(res, i):
    return ((res.y_pred == i) & (res.y_true != i)).sum() / (res.y_true != i).sum() # FPR = FP / N

def class_fnr(res, i):
    return ((res.y_pred != i) & (res.y_true == i)).sum() / (res.y_true == i).sum() # FNR = FN / P

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


## Combined Error Variance (CEV) and Symmetric Distance Error (SDE)
## C. Blakeney, G. Atkinson, N. Huish, Y. Yan, V. Metsis, Z. Zong, Measuring Bias and Fairness in Multiclass Classification, in: 2022 IEEE International Conference on Networking, Architecture and Storage (NAS), 2022: pp. 1–6. https://doi.org/10.1109/NAS55553.2022.9925287.


def combined_error_variance(res1, res2, vocab):
    n = len(vocab)
    fprs1 = [class_fpr(res1, i) for i in range(len(vocab))]
    fnrs1 = [class_fnr(res1, i) for i in range(len(vocab))]
    fprs2 = [class_fpr(res2, i) for i in range(len(vocab))]
    fnrs2 = [class_fnr(res2, i) for i in range(len(vocab))]
    dpos = [(fprs1[i] - fprs2[i]) / fprs2[i] for i in range(len(vocab))]
    dneg = [(fnrs1[i] - fnrs2[i]) / fnrs2[i] for i in range(len(vocab))]
    
    cve = np.mean([dist((np.mean(dpos), np.mean(dneg)), 
                  (dpos[i], dneg[i]))**2 for i in range(len(vocab))])
    
    return cve

def symmetric_distance_error(res1, res2, vocab):
    n = len(vocab)
    fprs1 = [class_fpr(res1, i) for i in range(len(vocab))]
    fnrs1 = [class_fnr(res1, i) for i in range(len(vocab))]
    fprs2 = [class_fpr(res2, i) for i in range(len(vocab))]
    fnrs2 = [class_fnr(res2, i) for i in range(len(vocab))]
    dfpr = [(fprs1[i] - fprs2[i]) / fprs2[i] for i in range(len(vocab))]
    dfnr = [(fnrs1[i] - fnrs2[i]) / fnrs2[i] for i in range(len(vocab))]

    sde = np.mean(np.abs(np.array(dfnr)-np.array(dfpr)))    
    return sde

## Intraclass Disparity and Overall Disparity
## I. Dominguez-Catena, D. Paternain, M. Galar, Assessing Demographic Bias Transfer from Dataset to Model: A Case Study in Facial Expression Recognition, in: Proceedings of the Workshop on Artificial Intelligence Safety 2022 (AISafety 2022), Vienna, Austria, 2022.
## https://doi.org/10.48550/arXiv.2205.10049.

def intraclass_disparity(dem, y_true, y_pred, targety):
    groups = dem.unique()
    recalls = []
    for g in groups:
        mask = dem == g
        mask2 = y_true == targety
        mask = mask & mask2
        recalls.append(sklearn.metrics.accuracy_score(y_true[mask], y_pred[mask]))
    if np.max(recalls) == 0:
        return 0
    else:
        return 1/(len(groups)-1)*np.sum([1 - r/np.max(recalls) for r in recalls])
    
def overall_disparity(dem, y_true, y_pred, vocab, agg=np.mean):
    ids = []
    for i, v in enumerate(vocab):
        ids.append(intraclass_disparity(dem, y_true, y_pred, i))
    return agg(ids)

## Novel proposals
## Adapted from the fairness definitions in P. Putzel, S. Lee, Blackbox Post-Processing for Multiclass Fairness, (2022). http://arxiv.org/abs/2201.04461


def confusion_matrix(df, normalize='true'):
    cm = sklearn.metrics.confusion_matrix(df.y_true, df.y_pred, normalize=normalize)
    return cm

def ttm_eq_odds(df, axis, groups):
    cms = []
    for g in groups:
        cm = confusion_matrix(df[df[axis] == g])
        cms.append(cm.flatten())
    cms = np.array(cms)
    return np.mean(cms.max(axis=0) - cms.min(axis=0))

def cm_eq_odds(df, axis, groups):
    cms = []
    for g in groups:
        cm = confusion_matrix(df[df[axis] == g])
        d = cm.diagonal()
        fdr = cm.sum(axis=0) - d
        fdr = fdr / np.sum(fdr)
        combi = np.concatenate([d.flatten(), fdr.flatten()])
        cms.append(combi)
    cms = np.array(cms)
    return np.mean(cms.max(axis=0) - cms.min(axis=0))

def m_eq_opportunity(df, axis, groups):
    cms = []
    for g in groups:
        cm = confusion_matrix(df[df[axis] == g])
        d = cm.diagonal()
        cms.append(d)

    cms = np.array(cms)
    return np.mean(cms.max(axis=0) - cms.min(axis=0))

def m_dem_parity(df, axis, groups):
    cms = []
    for g in groups:
        cm = confusion_matrix(df[df[axis] == g])
        p = cm.sum(axis=0)
        cms.append(p.flatten())

    cms = np.array(cms)
    return np.mean(cms.max(axis=0) - cms.min(axis=0))