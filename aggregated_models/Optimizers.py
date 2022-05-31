from aggregated_models.MyOptimizerUtils import (
    LineSearchWithOnlyGrad,
    LbfgsInvHessProduct,
)
import numpy as np


class WrapWithPrecond:
    """Wrapper around a model to make the variable changes x' := x *sqrt(Diag H)
    inputs: - model : instance of BaseAggModel  (whose parameters are "x" )
            - invDiagHessian:  np array of len nb Parameters.
                              (should be set to inverse diag of Hessian at optimum)
    This way the wrapped model hessian's diagonal at optimum is equal to identity.
    """

    def __init__(self, model, invDiagHessian):
        self.model = model
        self.invDiagHessian = invDiagHessian
        self.precond = np.sqrt(invDiagHessian)
        self.parameters = model.parameters / self.precond

    def setparameters(self, x):
        self.parameters = x
        self.model.setparameters(x * self.precond)

    def computeGradientAt(self, x):
        self.setparameters(x)
        g = self.model.computeGradientAt(self.model.parameters)
        return g * self.precond

    def computeLoss(self):
        return self.model.computeLoss()


def simpleGradientStep(
    model,
    nbiter,
    alpha=0.1,
    endIterCallback=None,
):
    for i in range(0, nbiter):
        print(f"simpleGradientStep iter={i}     ", end="\r")
        g = model.computeGradient()
        d = -g * model.computeInvHessianDiag()

        maxgrad = 2.0
        d[d > maxgrad] = maxgrad
        d[d < -maxgrad] = -maxgrad

        x = model.parameters
        xnew = x + alpha * d
        model.setparameters(xnew)
        if endIterCallback is not None:
            endIterCallback()

def lbfgs(
    self,
    nbiter=100,
    alpha=0.1,
    usePrecond=True,
    endIterCallback=None,
    nbSubDiagonals=10,
    verbose=True,
):

    if usePrecond:
        self = WrapWithPrecond(self, self.computeInvHessianDiagAtOptimum())

    def computeGrad(x):
        return self.computeGradientAt(x)

    lineSearch = LineSearchWithOnlyGrad(computeGrad, alpha)
    x = self.parameters
    g = computeGrad(x)
    invHess = LbfgsInvHessProduct(len(x), nbSubDiagonals)
    for i in range(0, nbiter):
        d = -invHess.multiplyByInvH(g)
        xnew, gnew = lineSearch.Search(x, d, g)
        dx = x - xnew
        dg = g - gnew
        invHess.update(dx, dg)
        x = xnew
        g = gnew
        lineSeatchMsg = lineSearch.getInfoStr()

        if endIterCallback is not None:
            endIterCallback()

        llh = self.computeLoss()
        if verbose:
            print(f"llh:{llh:.1E} " + lineSeatchMsg + f"  --", end="\r")
    if verbose:
        print("")


def Scipy_lbfgs(self, nbiter):
    #  Warapping scipy implem of lbfgs.
    #  might not work because loss is not correct !
    from scipy.optimize import minimize

    def myloss(x):
        self.setparameters(x)
        llh = self.computeLoss() * self.nbCoefs
        print("llh=", llh)
        return llh

    def mygrad(x):
        return self.computeGradientAt(x)

    optimresult = minimize(
        myloss,
        self.parameters,
        method="L-BFGS-B",
        jac=mygrad,
        options={"maxiter": nbiter},
    )
    self.setparameters(optimresult.x)
    print("")
