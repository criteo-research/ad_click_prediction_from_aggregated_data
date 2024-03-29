import numpy as np
import bisect
import random
from joblib import Parallel, delayed
from numba import jit
import numba


class VariableMRFParameters:
    def __init__(self, model):

        self.parameters = model.currentParams

        if not model.sampleFromPY0:
            self.parametersForPY1 = self.parameters.copy()
            for f in model.clickWeights:
                self.parametersForPY1[model.displayWeights[f].indices] += self.parametersForPY1[
                    model.clickWeights[f].indices
                ]
        # Copying things which only exist on some models
        try:  # for AggMRFModelWithAggPreds
            self.theta0 = model.theta0
            self.offsetNu = model.offsetNu
        except:
            pass

    @property
    def nu(self):
        return self.parameters[self.offsetNu :]


class ConstantMRFParameters:
    def __init__(
        self,
        nbSamples,
        nbParameters,
        sampleFromPY0,
        explosedDisplayWeights,
        explosedClickWeights,
        displayWeights,
        clickWeights,
        modalitiesByVarId,
        muIntercept,
        lambdaIntercept,
    ):
        self.nbSamples = nbSamples
        self.nbParameters = nbParameters
        self.sampleFromPY0 = sampleFromPY0
        self.explosedDisplayWeights = explosedDisplayWeights
        self.explosedClickWeights = explosedClickWeights
        self.displayWeights = displayWeights
        self.clickWeights = clickWeights
        self.modalitiesByVarId = modalitiesByVarId
        self.muIntercept = muIntercept
        self.lambdaIntercept = lambdaIntercept


@jit(nopython=True)
def mybisect(a: np.ndarray, x):
    """Similar to bisect.bisect() or bisect.bisect_right(), from the built-in library."""
    n = a.size
    if n == 1:
        return 0
    left = 0
    right = n
    while right > left + 1:
        current = int((left + right) / 2)
        if x >= a[current]:
            left = current
        else:
            right = current
    if x >= a[left]:
        return right
    return left


@jit(nopython=True)
def weightedSampleNUMBA(p):
    cp = np.cumsum(p)
    r = np.random.random() * cp[len(cp) - 1]
    return mybisect(cp, r)


def weightedSample(p):
    cp = np.cumsum(p)
    r = np.random.random() * cp[len(cp) - 1]
    return bisect.bisect(cp, r)


@jit(nopython=True)
def weightedSamplesNUMBA(M):
    return np.array([weightedSampleNUMBA(p) for p in M])


def weightedSamples(M):
    return np.array([weightedSample(p) for p in M])


# @jit(nopython=True) # Not working, cannot remember why.
def fastGibbsSample(
    exportedDisplayWeights,
    exportedClickWeights,
    modalitiesByVarId,
    paramsVector,
    x,
    nbsteps,
    y,
):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights
    (
        click_allcoefsv,
        click_allcoefsv2,
        click_alloffsets,
        click_allotherfeatureid,
        click_allmodulos,
    ) = exportedClickWeights

    x = x.transpose().copy()
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    nbsamples = x.shape[1]

    # Iterating on nbsteps steps
    for i in np.arange(0, nbsteps):

        # List of features ( one feature <=> its index in the arrays )
        f = np.arange(0, x.shape[0])
        #  Gibbs sampling may converge faster if the order in which we sample features is randomized.
        np.random.shuffle(f)

        # For each feature, ressample this feature conditionally to the other
        for varId in f:

            # data describing the different crossfeatures involving varId  in " K(x).mu"
            # Those things are arrays, of len the number of crossfeatures involving varId.
            disp_coefsv = allcoefsv[varId]
            disp_coefsv2 = allcoefsv2[varId]
            disp_offsets = alloffsets[varId]
            disp_otherfeatureid = allotherfeatureid[varId]
            disp_modulos = allmodulos[varId]
            # idem, but crossfeatures in " K(x).theta" part of the model
            click_coefsv = click_allcoefsv[varId]
            click_coefsv2 = click_allcoefsv2[varId]
            click_offsets = click_alloffsets[varId]
            click_otherfeatureid = click_allotherfeatureid[varId]
            click_modulos = click_allmodulos[varId]

            # array of possible modalities of varId
            modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
            # for each modality, compute P( modality | other features ) as exp( dotproduct)
            # initializing dotproduct
            mus = np.zeros((nbsamples, len(modalities)))
            lambdas = np.zeros((nbsamples, len(modalities)))

            # Computing the dotproducts
            #  For each crossfeature containing varId
            for varJ in np.arange(0, len(disp_coefsv)):
                modulo = disp_modulos[varJ]

                # let m a modality of feature varId, and m' a modality of the other feature
                #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
                # values of m' in the data
                modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
                # all possible modality m
                mods = modalities * disp_coefsv[varJ]
                # Computing crossfeatures
                # this is a matrix of shape (nbSamples, nbModalities of varId)
                crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
                # Adding crossfeature weight.
                mus += paramsVector[crossmods]

            for varJ in np.arange(0, len(click_coefsv)):
                modulo = click_modulos[varJ]

                modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
                mods = modalities * click_coefsv[varJ]
                crossmods = (np.add.outer(modsJ, mods) % modulo) + click_offsets[varJ]
                # mus += paramsVector[crossmods] # buggycode

                lambdas += paramsVector[crossmods]

            mus = np.exp(mus)
            lambdas = np.exp(lambdas)

            mus = mus * ((lambdas.transpose() * y + 1 - y)).transpose()
            # mus = mus * ( 1  +  lambdas) # buggycode
            # Sampling now modality of varId
            varnew = weightedSamples(mus)
            # updating the samples
            x[varId] = varnew

    return x.transpose()


# Sampling X from P( X | Y = 0, mu)
# - Should be twice faster than sampling from P(X,Y) (no need to compute P(Y|X) part of the model)
# - We can later use importance weighting to correct for this (and not so different anyway)


def fastGibbsSampleFromPY0(exportedDisplayWeights, modalitiesByVarId, paramsVector, x, nbsteps):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights

    x = x.transpose().copy()
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    nbsamples = x.shape[1]

    # Iterating on nbsteps steps
    for i in np.arange(0, nbsteps):

        # List of features ( one feature <=> its index in the arrays )
        f = np.arange(0, x.shape[0])
        #  Gibbs sampling may converge faster if the order in which we sample features is randomized.
        np.random.shuffle(f)

        # For each feature, ressample this feature conditionally to the other
        for varId in f:

            # data describing the different crossfeatures involving varId  in " K(x).mu"
            # Those things are arrays, of len the number of crossfeatures involving varId.
            disp_coefsv = allcoefsv[varId]
            disp_coefsv2 = allcoefsv2[varId]
            disp_offsets = alloffsets[varId]
            disp_otherfeatureid = allotherfeatureid[varId]
            disp_modulos = allmodulos[varId]

            # array of possible modalities of varId
            modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
            # for each modality, compute P( modality | other features ) as exp( dotproduct)
            # initializing dotproduct
            mus = np.zeros((nbsamples, len(modalities)))

            # Computing the dotproducts
            #  For each crossfeature containing varId
            for varJ in np.arange(0, len(disp_coefsv)):
                modulo = disp_modulos[varJ]
                # let m a modality of feature varId, and m' a modality of the other feature
                #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
                # values of m' in the data
                modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
                # all possible modality m
                mods = modalities * disp_coefsv[varJ]
                # Computing crossfeatures
                # this is a matrix of shape (nbSamples, nbModalities of varId)
                crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
                # Adding crossfeature weight.
                mus += paramsVector[crossmods]

            mus = np.exp(mus)
            # Sampling now modality of varId
            varnew = weightedSamples(mus)
            # updating the samples
            x[varId] = varnew
    return x.transpose()


def cos(g, g2):
    return g.dot(g2) / np.sqrt(g.dot(g) * g2.dot(g2))


# @jit(nopython=True)
def computeRaoBlackwellisedExpectations(
    exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, paramsVector, x, py
):

    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights
    (
        click_allcoefsv,
        click_allcoefsv2,
        click_alloffsets,
        click_allotherfeatureid,
        click_allmodulos,
    ) = exportedClickWeights

    x = x.transpose()
    nbsamples = x.shape[1]

    results = np.zeros(len(paramsVector))

    # List of features ( one feature <=> its index in the arrays )
    f = np.arange(0, x.shape[0])
    #  Gibbs sampling may converge faster if the order in which we sample features is randomized.

    # For each feature, ressample this feature conditionally to the other
    for varId in f:

        # data describing the different crossfeatures involving varId  in " K(x).mu"
        # Those things are arrays, of len the number of crossfeatures involving varId.
        disp_coefsv = allcoefsv[varId]
        disp_coefsv2 = allcoefsv2[varId]
        disp_offsets = alloffsets[varId]
        disp_otherfeatureid = allotherfeatureid[varId]
        disp_modulos = allmodulos[varId]

        # idem, but crossfeatures in " K(x).theta" part of the model
        click_coefsv = click_allcoefsv[varId]
        click_coefsv2 = click_allcoefsv2[varId]
        click_offsets = click_alloffsets[varId]
        click_otherfeatureid = click_allotherfeatureid[varId]
        click_modulos = click_allmodulos[varId]

        # array of possible modalities of varId
        modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
        # for each modality, compute P( modality | other features ) as exp( dotproduct)
        # initializing dotproduct
        mus = np.zeros((nbsamples, len(modalities)))
        lambdas = np.zeros((nbsamples, len(modalities)))

        # Computing the dotproducts
        #  For each crossfeature containing varId
        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = disp_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
            # Adding crossfeature weight.
            mus += paramsVector[crossmods]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            # all possible modality m
            mods = modalities * click_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (np.add.outer(modsJ, mods) % modulo) + click_offsets[varJ]
            lambdas += paramsVector[crossmods]

        mus = np.exp(mus)
        mus = mus / mus.sum(axis=1)[:, None]

        currentLambdas = lambdas[np.arange(len(lambdas)), x[varId]]
        lambdas -= currentLambdas[:, None]
        lambdas = np.exp(lambdas) * mus * py[:, None]

        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = click_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)

            crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
            for i in range(0, nbsamples):
                results[crossmods[i]] += mus[i]
            # for i in range(0,nbsamples):
            #    xmods = mods  +  modsJ[i] + disp_offsets[varJ]
            #    results[xmods] += mus[i]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            mods = modalities * click_coefsv[varJ]
            for i in range(0, nbsamples):
                xmods = (mods + modsJ[i]) % modulo + click_offsets[varJ]
                results[xmods] += lambdas[i]

        # lambdas = np.exp(lambdas)
        # mus = mus * (( lambdas.transpose() * y + 1-y )).transpose()
        # mus = mus * ( 1  +  lambdas) # buggycode
        # Sampling now modality of varId

        # varnew =  weightedSamples(mus)
        # updating the samples
        # x[varId] = varnew

    return results


@jit(nopython=True)
def computeRaoBlackwellisedExpectations_numba(
    exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, paramsVector, x, py
):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights
    (
        click_allcoefsv,
        click_allcoefsv2,
        click_alloffsets,
        click_allotherfeatureid,
        click_allmodulos,
    ) = exportedClickWeights

    x = x.transpose()
    nbsamples = x.shape[1]

    results = np.zeros(len(paramsVector))

    # List of features ( one feature <=> its index in the arrays )
    f = np.arange(0, x.shape[0])
    #  Gibbs sampling may converge faster if the order in which we sample features is randomized.

    # For each feature, ressample this feature conditionally to the other
    for varId in f:

        # data describing the different crossfeatures involving varId  in " K(x).mu"
        # Those things are arrays, of len the number of crossfeatures involving varId.
        disp_coefsv = allcoefsv[varId]
        disp_coefsv2 = allcoefsv2[varId]
        disp_offsets = alloffsets[varId]
        disp_otherfeatureid = allotherfeatureid[varId]
        disp_modulos = allmodulos[varId]

        # idem, but crossfeatures in " K(x).theta" part of the model
        click_coefsv = click_allcoefsv[varId]
        click_coefsv2 = click_allcoefsv2[varId]
        click_offsets = click_alloffsets[varId]
        click_otherfeatureid = click_allotherfeatureid[varId]
        click_modulos = allmodulos[varId]

        # array of possible modalities of varId
        modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
        # for each modality, compute P( modality | other features ) as exp( dotproduct)
        # initializing dotproduct
        mus = np.zeros((nbsamples, len(modalities)))
        lambdas = np.zeros((nbsamples, len(modalities)))

        # Computing the dotproducts
        #  For each crossfeature containing varId
        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = disp_modulos[varJ]
            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (AddOuter(modsJ, mods) % modulo) + disp_offsets[varJ]
            # Adding crossfeature weight.
            mus += getVectorValuesAtIndex(paramsVector, crossmods)
            # mus += paramsVector[crossmods]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]
            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            # all possible modality m
            mods = modalities * click_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (AddOuter(modsJ, mods) % modulo) + click_offsets[varJ]
            lambdas += getVectorValuesAtIndex(paramsVector, crossmods)

        mus = np.exp(mus)

        for i in range(0, mus.shape[0]):
            s = mus[i, :].sum()
            mus[i, :] /= s
        # mus = (mus / mus.sum(axis=1)[:,None]  )

        for i in range(0, mus.shape[0]):
            currentModality = x[varId, i]
            currentLambda = lambdas[i, currentModality]
            lambdas[i, :] -= currentLambda

        # currentLambdas = lambdas[np.arange(len(lambdas)), x[varId]]
        # lambdas -= currentLambdas[:,None]
        # lambdas = np.exp( lambdas) * mus  * py[:,None]
        lambdas = np.exp(lambdas) * mus
        for i in range(0, mus.shape[0]):
            lambdas[i, :] *= py[i]
        # lambdas *=  py[:,None]

        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = disp_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)

            crossmods = (AddOuter(modsJ, mods) % modulo) + disp_offsets[varJ]
            for i in range(0, nbsamples):
                results[crossmods[i]] += mus[i]
            # for i in range(0,nbsamples):
            #    xmods = mods  +  modsJ[i] + disp_offsets[varJ]
            #    results[xmods] += mus[i]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            mods = modalities * click_coefsv[varJ]
            for i in range(0, nbsamples):
                xmods = (mods + modsJ[i]) % modulo + click_offsets[varJ]
                results[xmods] += lambdas[i]

        # lambdas = np.exp(lambdas)
        # mus = mus * (( lambdas.transpose() * y + 1-y )).transpose()
        # mus = mus * ( 1  +  lambdas) # buggycode
        # Sampling now modality of varId

        # varnew =  weightedSamples(mus)
        # updating the samples
        # x[varId] = varnew

    return results


@jit(nopython=True)
def AddOuter(x, y):
    r = np.zeros((len(y), len(x)), dtype="int32")
    r += x
    r = r.transpose()
    r += y
    return r


@jit(nopython=True)
def getVectorValuesAtIndex(x, indexingMatrix):
    r = np.zeros(indexingMatrix.shape, dtype="float64")
    for i in range(0, indexingMatrix.shape[0]):
        for j in range(0, indexingMatrix.shape[1]):
            r[i, j] = x[indexingMatrix[i, j]]
    return r


@jit(nopython=True)
def gibbsOneSampleFromPY0(exportedDisplayWeights, modalitiesByVarId, paramsVector, x, nbsteps):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights

    # Iterating on nbsteps steps
    for i in np.arange(0, nbsteps):

        # List of features ( one feature <=> its index in the arrays )
        f = np.arange(0, len(x))
        #  Gibbs sampling may converge faster if the order in which we sample features is randomized.
        np.random.shuffle(f)

        # For each feature, ressample this feature conditionally to the other
        for varId in f:

            # data describing the different crossfeatures involving varId  in " K(x).mu"
            # Those things are arrays, of len the number of crossfeatures involving varId.
            disp_coefsv = allcoefsv[varId]
            disp_coefsv2 = allcoefsv2[varId]
            disp_offsets = alloffsets[varId]
            disp_otherfeatureid = allotherfeatureid[varId]
            disp_modulos = allmodulos[varId]

            # array of possible modalities of varId
            modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
            # for each modality, compute P( modality | other features ) as exp( dotproduct)
            # initializing dotproduct
            mus = np.zeros(len(modalities))

            # Computing the dotproducts
            #  For each crossfeature containing varId
            for varJ in np.arange(0, len(disp_coefsv)):
                modulo = disp_modulos[varJ]
                # let m a modality of feature varId, and m' a modality of the other feature
                #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
                # values of m' in the data
                modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
                # all possible modality m
                mods = modalities * disp_coefsv[varJ]
                # Computing crossfeatures
                # this is a matrix of shape (nbSamples, nbModalities of varId)
                crossmods = ((modsJ + mods) % modulo) + disp_offsets[varJ]
                # Adding crossfeature weight.
                mus += paramsVector[crossmods]

            mus = np.exp(mus)
            # Sampling now modality of varId
            varnew = weightedSampleNUMBA(mus)
            # updating the samples
            x[varId] = varnew
    return x


@jit
def oneHotEncode(x, explosedDisplayWeights):
    f = np.arange(0, len(x))
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = explosedDisplayWeights
    nb_weights = len(allcoefsv)

    proj = []

    # For each feature, ressample this feature conditionally to the other
    for varId in f:

        # data describing the different crossfeatures involving varId  in " K(x).mu"
        # Those things are arrays, of len the number of crossfeatures involving varId.
        disp_coefsv = allcoefsv[varId]
        disp_coefsv2 = allcoefsv2[varId]
        disp_offsets = alloffsets[varId]
        disp_otherfeatureid = allotherfeatureid[varId]
        disp_modulos = allmodulos[varId]

        # Computing the dotproducts
        #  For each crossfeature containing varId
        for varJ in np.arange(0, len(disp_coefsv)):

            modulo = disp_modulos[varJ]
            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = x[varId] * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = ((modsJ + mods) % modulo) + disp_offsets[varJ]
            # Adding crossfeature weight.
            proj.append(crossmods)

    return np.unique(np.array(proj, np.int32))


@jit(nopython=True)
def batchVariables(featureSizes, maxNbModalities):
    shuffledFeatures = np.arange(0, len(featureSizes))
    np.random.shuffle(shuffledFeatures)
    batches = [
        [shuffledFeatures[0]]
    ]  # Putting first variable in first batch to avoid instanciating empty list (makes numba unhappy)
    varids = batches[0]
    nbvarsInBatch = 1
    nbmodalities = featureSizes[shuffledFeatures[0]]
    for fid in shuffledFeatures[1:]:
        nbmodalities *= featureSizes[fid]
        nbvarsInBatch += 1
        if nbmodalities > maxNbModalities and nbvarsInBatch > 1:
            varids = [fid]
            nbvarsInBatch = 1
            nbmodalities = featureSizes[fid]
            batches.append(varids)
        else:
            varids.append(fid)
            nbvarsInBatch += 1

    # print(batches , maxNbModalities , featureSizes)
    return batches


@jit(nopython=True)
def buildAllModalities(varIds, modalitiesByVarId):

    totalnbmodalities = 1
    totalnbmodalitiesBeforeVar = []

    for j in varIds:
        totalnbmodalitiesBeforeVar.append(totalnbmodalities)
        totalnbmodalities *= len(modalitiesByVarId[j])

    allmodalities = np.zeros((len(varIds), totalnbmodalities), dtype=numba.int32)
    j = 0
    for varId in varIds:
        modalities = modalitiesByVarId[varId]
        nbmodalities = len(modalities)
        nbcrossmodsBefore = totalnbmodalitiesBeforeVar[j]
        k = 0
        for nbVectorCopies in range(0, int(totalnbmodalities / nbmodalities / nbcrossmodsBefore)):
            for m in modalities:
                for nbcopies in range(0, nbcrossmodsBefore):
                    allmodalities[j, k] = m
                    k += 1
        j += 1
    return allmodalities


@jit(nopython=True)
def blockedGibbsSampler_PY0(exportedDisplayWeights, modalitiesByVarId, paramsVector, x, nbsteps, maxNbModalities):

    if maxNbModalities <= 1:
        return gibbsOneSampleFromPY0(exportedDisplayWeights, modalitiesByVarId, paramsVector, x, nbsteps)

    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights

    for step in np.arange(0, nbsteps):

        batches = batchVariables([len(x) for x in modalitiesByVarId], maxNbModalities)

        for featuresBatch in batches:
            allmodalities = buildAllModalities(featuresBatch, modalitiesByVarId)

            nbmodalities = allmodalities.shape[1]
            mus = np.zeros(nbmodalities)

            varIndexInBatch = -1
            for varId in featuresBatch:
                varIndexInBatch += 1
                # data describing the different crossfeatures involving varId  in " K(x).mu"
                # Those things are arrays, of len the number of crossfeatures involving varId.
                disp_coefsv = allcoefsv[varId]
                disp_coefsv2 = allcoefsv2[varId]
                disp_offsets = alloffsets[varId]
                disp_otherfeatureid = allotherfeatureid[varId]
                disp_modulos = allmodulos[varId]

                # array of possible modalities of varId
                modalities = modalitiesByVarId[varId]
                # Should be 0,1,2 ... NbModalities
                # for each modality, compute P( modality | other features ) as exp( dotproduct)
                # initializing dotproduct
                # Computing the dotproducts
                #  For each crossfeature containing varId
                for j in np.arange(0, len(disp_coefsv)):

                    varJ = disp_otherfeatureid[j]

                    if j == 0:
                        varJ = varId  # HACK because allotherfeatureid contains always 0 at index 0

                    if varJ in featuresBatch:

                        if varJ < varId:
                            continue  # avoid applying crossweight twice

                        indexOfVarJ_inBatch = 0
                        while featuresBatch[indexOfVarJ_inBatch] != varJ:
                            indexOfVarJ_inBatch += 1

                        modulo = disp_modulos[j]
                        modsJ = allmodalities[indexOfVarJ_inBatch] * disp_coefsv2[j]
                        mods = allmodalities[varIndexInBatch] * disp_coefsv[j]
                        # Computing crossfeatures
                        # this is a matrix of shape (nbSamples, nbModalities of varId)
                        crossmods = ((modsJ + mods) % modulo) + disp_offsets[j]
                        # Adding crossfeature weight.
                        mus += paramsVector[crossmods]

                    else:
                        modulo = disp_modulos[j]
                        # let m a modality of feature varId, and m' a modality of the other feature
                        #  Value of the crossfeature is " m *  disp_coefsv[j] + m' * disp_coefsv2[varJ]  "
                        # values of m' in the data
                        modsJ = x[disp_otherfeatureid[j]] * disp_coefsv2[j]
                        # all possible modality m
                        mods = allmodalities[varIndexInBatch] * disp_coefsv[j]
                        # Computing crossfeatures
                        # this is a matrix of shape (nbSamples, nbModalities of varId)
                        crossmods = ((modsJ + mods) % modulo) + disp_offsets[j]
                        # Adding crossfeature weight.
                        mus += paramsVector[crossmods]

            mus = np.exp(mus)
            # Sampling now modality of varId
            varnew = weightedSampleNUMBA(mus)
            # updating the samples

            for indexInBatch, fid in enumerate(featuresBatch):
                x[fid] = allmodalities[indexInBatch][varnew]
    return x


# Sampling from  P(X=x  |Y=0) := exp( K(x).mu + f(x) K(x).vu   ) /Z
#  avec f(x) :=  LinkFunction( K(x).theta0 )
@jit(nopython=True)
def fastGibbsSampleFromPY0_withAggPredictions(
    exportedDisplayWeights,
    modalitiesByVarId,
    paramsVector,
    params_f0,
    params_vu,
    x,
    kx_dot_params_f0,
    kx_dot_vu,
    nbsteps,
    linkFunctionId,  #  0 -> Identity  , 1 -> sigmoid, 2 -> exp
):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights

    # List of features ( one feature <=> its index in the arrays )
    f = np.arange(0, len(x))

    # Iterating on nbsteps steps
    for i in np.arange(0, nbsteps):

        #  Gibbs sampling may converge faster if the order in which we sample features is randomized.
        np.random.shuffle(f)

        # For each feature, ressample this feature conditionally to the other
        for varId in f:

            # data describing the different crossfeatures involving varId  in " K(x).mu"
            # Those things are arrays, of len the number of crossfeatures involving varId.
            disp_coefsv = allcoefsv[varId]
            disp_coefsv2 = allcoefsv2[varId]
            disp_offsets = alloffsets[varId]
            disp_otherfeatureid = allotherfeatureid[varId]
            disp_modulos = allmodulos[varId]

            # array of possible modalities of varId
            modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
            # for each modality, compute P( modality | other features ) as exp( dotproduct)
            # initializing dotproduct
            mus = np.zeros(len(modalities))
            vus = np.zeros(len(modalities))
            theta0s = np.zeros(len(modalities))

            # Computing the dotproducts
            #  For each crossfeature containing varId
            for varJ in np.arange(0, len(disp_coefsv)):

                modulo = disp_modulos[varJ]
                # let m a modality of feature varId, and m' a modality of the other feature
                #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
                # values of m' in the data
                modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
                # all possible modality m
                mods = modalities * disp_coefsv[varJ]
                # Computing crossfeatures
                ## crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
                crossmods = (modsJ + mods) % modulo + disp_offsets[varJ]
                # Adding crossfeature weight.
                mus += paramsVector[crossmods]
                vus += params_vu[crossmods]
                theta0s += params_f0[crossmods]

            # current modality
            xi = x[varId]

            vus = vus - vus[xi] + kx_dot_vu
            theta0s = theta0s - theta0s[xi] + kx_dot_params_f0
            # applying link function

            if linkFunctionId == 0:
                f0s = theta0s
            elif linkFunctionId == 1:
                f0s = 1 / (1 + np.exp(-theta0s))  # sigmoid of K(x).theta0
            elif linkFunctionId == 2:
                f0s = np.exp(theta0s)

            probas = np.exp(mus + f0s * vus)

            # Sampling now modality of varId
            varnew = weightedSampleNUMBA(probas)
            # updating the sample
            x[varId] = varnew
            # and updating precomputed dotproducts
            kx_dot_vu = vus[varnew]
            kx_dot_params_f0 = theta0s[varnew]

    return x


from numba import jit


@jit(nopython=True)
def ComputeKernel_OneSample(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector):
    mus = np.zeros(len(modalities))
    for varJ in np.arange(0, len(coefsv)):
        modulo = modulos[varJ]
        modsJ = x[otherfeatureid[varJ]] * coefsv2[varJ]
        mods = modalities * coefsv[varJ]
        crossmods = (modsJ + mods) % modulo + offsets[varJ]
        mus += paramsVector[crossmods]
    p = np.exp(mus)
    p = p / p.sum()
    return p


@jit(nopython=True)
def SampleFromKernel(samples, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector):
    nbsamples = samples.shape[0]
    y = np.zeros(nbsamples, dtype=np.int32)
    for i, x in enumerate(samples):
        y[i] = SampleOneFromKernel(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
    return y


@jit(nopython=True)
def SampleOneFromKernel(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector):
    probas = ComputeKernel_OneSample(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
    varnew = weightedSampleNUMBA(probas)
    return varnew


@jit(nopython=True)
def ComputeKernel(samples, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector):
    nbsamples = samples.shape[0]
    mus = np.zeros((nbsamples, len(modalities)))
    for i, x in enumerate(samples):
        mus[i] = ComputeKernel_OneSample(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
    return mus


@jit(nopython=True)
def AddExpectedK_oneSample(pData, x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector):

    p = ComputeKernel_OneSample(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
    for varJ in np.arange(0, len(coefsv)):
        modulo = modulos[varJ]
        modsJ = x[otherfeatureid[varJ]] * coefsv2[varJ]
        mods = modalities * coefsv[varJ]
        crossmods = (modsJ + mods) % modulo + offsets[varJ]
        pData[crossmods] += p


@jit(nopython=True)
def ExpectedK(samples, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector):
    pData = np.zeros(len(paramsVector))
    for x in samples:
        AddExpectedK_oneSample(pData, x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
    return pData
