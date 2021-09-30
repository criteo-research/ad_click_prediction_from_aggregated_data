from aggregated_models.mrf_helpers import *
import numpy as np
import matplotlib.pyplot as plt


def ViewXdistribs(model, fs=None, x=None, nbshow=3, viewY=False):
    if fs is None:
        fs = [x for x in model.displayWeights]
    if x is None:
        try:
            x = np.array(model.samples.rddSamples.take(100_000))
        except:
            x = np.array(model.samples.columns.transpose())

    preds = None
    if viewY:
        preds = model.getPredictionsVector(model.samples)

    results = {}
    for f in fs:
        results[f] = ViewXdistrib(model, f, x, doprint=False, preds=preds, viewY=viewY)
    topkeys = sorted(results, key=results.get)[-nbshow:][::-1]
    for f in topkeys:
        print(f"{f} , {results[f]}")
    for f in topkeys:
        results[f] = ViewXdistrib(model, f, x, doprint=True, preds=preds, viewY=viewY)


def ViewXdistrib(model, f, x, preds=None, doprint=True, viewY=False):
    w = model.displayWeights[f]
    wc = model.clickWeights[f]

    if viewY:
        d = model.Data[w.indices]
        p = preds[w.indices]
        dc = model.Data[wc.indices]
        pc = preds[wc.indices]

        dc_corrected = dc * np.minimum((p + 1) / (d + 1), 1)
        pc_corrected = pc * np.minimum((d + 1) / (p + 1), 1)
        return compareDistribs(dc_corrected, pc_corrected, f, doprint=doprint)

    else:
        d = model.Data[w.indices] - model.Data[wc.indices]
        n = x.shape[0]
        y = np.ones(n)
        p = w.feature.Project_(x.transpose(), y)

    return compareDistribs(d, p, f, doprint=doprint)


def compareDistribs(d, p, f="", doprint=True):
    p = p / sum(p)
    d = d / sum(d)

    nozerosinds = np.where((d + p) > 0)
    p = p[nozerosinds]
    d = d[nozerosinds]

    n = 100_000
    z = np.random.multinomial(n, d)
    z = z / sum(z)

    ind = np.argsort(d)
    ind = ind[::-1]

    area_p = sum(np.cumsum(p[ind])) / len(p)
    area_d = sum(np.cumsum(d[ind])) / len(p)
    area_z = sum(np.cumsum(z[ind])) / len(p)

    if doprint:
        plt.figure()
        plt.plot(n * np.cumsum(p[ind]), "r")
        plt.plot(n * np.cumsum(d[ind]), "k")
        plt.plot(n * np.cumsum(z[ind]), "g")
        plt.title(f + " " + str(area_p - area_z))
    return np.abs(area_p - area_z)


def computeAreaDiff(d, p):
    nozerosinds = np.where((d + p) > 0)
    p = p[nozerosinds]
    d = d[nozerosinds]
    p = p / sum(p)
    d = d / sum(d)
    ind = np.argsort(d)
    ind = ind[::-1]
    area_p = sum(np.cumsum(p[ind])) / len(p)
    area_d = sum(np.cumsum(d[ind])) / len(p)
    return np.abs(area_p - area_d)


def getSamplesAfterGibbs(model, x=None, nbiters=10, maxNbModalities=100_000):
    if x is None:
        x = np.array(model.samples.rddSamples.take(10_000))
    exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, paramsVector = model.exportWeightsAll()
    ss = model.samples.sparkSession
    rdd = ss.sparkContext.parallelize(list(zip(x, list(range(0, x.shape[0])))), 200)

    def myfun(s):
        return (
            blockedGibbsSampler_PY0(
                exportedDisplayWeights, modalitiesByVarId, paramsVector, s[0], nbiters, maxNbModalities
            ),
            s[1],
        )

    a = rdd.map(myfun).collect()
    x2 = np.array([s[0] for s in sorted(a, key=lambda s: s[1])])
    return x, x2


def checkGibbs(model, x=None, nbiters=10, maxNbModalities=100_000):
    x, x2 = getSamplesAfterGibbs(model, x, nbiters, maxNbModalities)
    for f in range(x.shape[1]):
        xf = x[:, f].copy()
        xf2 = x2[:, f].copy()

        countEqual = len(np.where(xf == xf2)[0])
        np.random.shuffle(xf)
        expectedEqual = len(np.where(xf == xf2)[0])
        print(model.features[f], countEqual, expectedEqual)


def worstFeature(model):
    preds = model.getPredictionsVector(model.samples)
    r = {}
    for f in model.displayWeights:
        w = model.displayWeights[f]
        d = model.Data[w.indices]
        p = preds[w.indices]
        r[f] = computeAreaDiff(d, p)
    f = sorted(r, key=r.get)[-1:][0]
    print(" worstFeature ", f, r[f])
    return f


def PoissonLLH(l, n, sigma=50, nbsamples=None):  # l poisson parameter, n observed nb samples
    # P_l(n) = exp( -l ) l^n  /n!
    # log P_l(n) ~= -l +n log(l) + n - n log(n)    (Stirling: 1/n! ~= n^n.exp(-n) )
    if nbsamples is not None:
        l = l * nbsamples / l.sum()
        n = n * nbsamples / n.sum()
    llh = n - l + (n + sigma) * (np.log(l + sigma) - np.log(n + sigma))
    return llh


def EPoissonLLH(l, sigma=50, nbsamples=1_000_000):
    lsum = l.sum()
    z = np.random.multinomial(nbsamples, l / lsum)
    return PoissonLLH(l, z, sigma, 1_000_000)


def PoissonScore(a, b, sigma, nbsamples):
    llh = PoissonLLH(a, b, sigma, nbsamples).sum()
    zllh = EPoissonLLH(a, sigma, nbsamples).sum()
    return zllh - llh


def worstClickFeature(model):
    preds = model.getPredictionsVector(model.samples)
    r = {}
    for f in model.clickWeights:
        w = model.displayWeights[f]
        wc = model.clickWeights[f]
        d = model.Data[w.indices]
        c = model.Data[wc.indices]
        p = preds[w.indices]
        pc = preds[wc.indices]

        c_corrected = c * np.minimum((p + 1) / (d + 1), 1)
        pc_corrected = pc * np.minimum((d + 1) / (p + 1), 1)

        r[f] = PoissonScore(c_corrected, pc_corrected, 50, 1_000_000)
    f = sorted(r, key=r.get)[-1:][0]
    print(" worst click Feature ", f, r[f])

    return f


def blockCooGradStep(model, f="hash_0&hash_10", clicks=False, alpha=0.5, curv=100.0):

    w = model.displayWeights[f]
    if clicks:
        w = model.clickWeights[f]

    g = model.computeGradient()

    p = model.getPredictionsVector(model.samples)
    hinv = 1 / (curv * 2 + model.Data + p)
    d2 = -g * hinv

    model.parameters[w.indices] = model.parameters[w.indices] + d2[w.indices] * alpha
    model.setparameters(model.parameters)


from aggregated_models.featuremappings import *


def buildfs(fs, model):
    def getfid(f):
        return model.displayWeights[f].feature._fid

    if len(fs) == 3:
        return TripletFeaturesMapping(fs[0], fs[1], fs[2], getfid(fs[0]), getfid(fs[1]), getfid(fs[2]))
    if len(fs) == 2:
        return model.displayWeights[sorted(fs)[0] + "&" + sorted(fs)[1]].feature
        return CrossFeaturesMapping(fs[0], fs[1], getfid(fs[0]), getfid(fs[1]))
    if len(fs) == 1:
        return model.displayWeights[fs[0]].feature
