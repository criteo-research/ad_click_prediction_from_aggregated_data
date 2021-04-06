import numpy as np
from aggregated_models.featuremappings import (
    CrossFeaturesMapping,
    SingleFeatureMapping,
)
from aggregated_models import featureprojections
from aggregated_models import Optimizers
from aggregated_models.baseaggmodel import BaseAggModel

# Logistic regression model from categorical features.
#  It is trained from:
#   - the aggregated labels
#   - the full event level dataset without labels.
# Note that those data are sufficient to fully train a logistic model.
# Tests show there is no difference in perfomances with other implentation of logistic regression.
# Side note:  this was implemented to begin with as a way to test BaseAggModel,
# it was finally faster than the other logistic regression I has, and so was used in most tests.


class AggLogistic(BaseAggModel):
    def __init__(self, aggdata, features, clicksCfs="*&*", regulL2=1.0):
        super().__init__(aggdata, features)
        self.regulL2 = regulL2
        self.clicksCfs = featureprojections.parseCFNames(self.features, clicksCfs)
        self.setProjections()  # building all weights and projections now
        self.setWeights()
        self.initParameters()
        self.nbCoefs = sum([w.feature.Size for w in self.clickWeights.values()])
        self.Data = self.getAggDataVector(self.clickWeights, self.clickProjections)

    def setProjections(self):
        fs = self.features + self.clicksCfs
        self.clickProjections = {var: self.aggdata.aggClicks[var] for var in fs}

    def setWeights(self):
        featuresAndCfs = self.features + self.clicksCfs
        self.clickWeights, offset = self.prepareWeights(featuresAndCfs)
        self.parameters = np.zeros(offset)

    def setDisplays(self, train):
        self.samples = self.transformDf(
            train[self.aggdata.features]
        ).values.transpose()  # keeping all features from aggdata to avoid breaking features indexing

    def initParameters(self):
        nbclicks = self.aggdata.Nbclicks
        nbdisplays = self.aggdata.Nbdisplays
        self.lambdaIntercept = np.log(nbclicks / (nbdisplays - nbclicks))

    def prepareFit(self, df):
        self.setDisplays(df)
        self.update()

    def predictDFinternal(self, df, pred_col_name: str):
        df["lambda"] = self.dotproductsOnDF(self.clickWeights, df) + self.lambdaIntercept
        df[pred_col_name] = 1.0 / (1.0 + np.exp(-df["lambda"]))
        return df

    def predictinternal(self):
        lambdas = self.dotproducts(self.clickWeights, self.samples)
        lambdas += self.lambdaIntercept
        self.pclick = 1.0 / (1.0 + np.exp(-lambdas))

    def update(self):
        self.predictinternal()

    def getPredictionsVector(self):
        x = self.parameters * 0
        for w in self.clickWeights.values():
            x[w.indices] = w.feature.Project_(self.samples, self.pclick)
        return x

    # Computing approx LLH, (not true LLH)
    def computeLoss(self, epsilon=1e-12):
        llh = self.computeLossNoRegul(epsilon)
        regul = (self.parameters * self.parameters).sum() * self.regulL2
        return (llh + regul) / self.nbCoefs

    #  "approx" loss.
    def computeLossNoRegul(self, epsilon=1e-12):
        preds = self.getPredictionsVector()
        llh = -(self.Data * np.log(preds + epsilon) - preds).sum()
        llh += (self.Data * np.log(self.Data + epsilon) - self.Data).sum()
        return (llh) / self.nbCoefs

    # grad of "exact" loss
    def computeGradient(self):  # grad of loss
        gradient = -self.Data + self.getPredictionsVector()
        gradient += 2 * self.parameters * self.regulL2
        self.normgrad = sum(gradient * gradient)
        return gradient

    def computeInvHessianDiagAtOptimum(self):  # grad of loss
        return 1 / (self.regulL2 * 2 + self.Data)

    def computeInvHessianDiag(self, alpha=0.9):  # grad of loss
        preds = self.getPredictionsVector()
        preds = preds * alpha + self.Data * (1 - alpha)  # averaging with value at optimum
        return 1 / (self.regulL2 * 2 + preds)

    def fit(self, train, nbIter=50, verbose=False):
        try:
            self.samples
        except Exception:
            self.prepareFit(train)
        Optimizers.lbfgs(self, nbiter=nbIter, alpha=0.01, verbose=verbose)

    def computeExactLoss(self, df):
        p = self.predictDF(df, "pclick").pclick.values
        y = df[self.label].values
        avgy = sum(y) / len(y)
        llh = -sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        llh -= -sum(y * np.log(avgy) + (1.0 - y) * np.log(1.0 - avgy))
        regul = sum(self.parameters * self.parameters) * self.regulL2
        return (regul + llh) / self.nbCoefs

    def __repr__(self):
        return f"logitic({self.features},λ={self.regulL2:.1E})"
