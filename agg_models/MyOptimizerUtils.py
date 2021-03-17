import numpy as np
from numpy import array, asarray, float64, zeros
from scipy.sparse.linalg import LinearOperator
import math


class LineSearchWithOnlyGrad:
    """Line search method for minimizing convex function with gradient descent, where only gradient is computable (ie function f itself is *not* available)
    Idea:  ensure that f(x+dx) < f(x) but looking for a step dx such that grad(x+dx).dx is still negative.
    Parameters:
     - gradFun: array -> array  Function computing grad(x).
     - alpha: initial step size
    """

    def __init__(self, gradFun, alpha=0.01):  # function returning gradient at x
        self.gradFun = gradFun
        self.alpha = alpha
        self.nbCalls = 0
        self.lastNbTry = 0

    def Search(self, x0, d, g0=None):
        """
        x0: current parameter estimate. array of size n.
        d : descent direction (array of size n)
        g0: gradient at x0 (if already computed
        returning x: new parameter estimate, g: gradient at x
        """
        self.nbCalls += 1
        if g0 is None:
            g0 = self.gradFun(x0)
        dg0 = d.dot(g0)

        if dg0 >= 0:
            print(f"Linesearch error: not a descent direction {dg0:.1E} ")
            return
        alpha = self.alpha

        x = x0 + d * alpha
        g = self.gradFun(x)
        dg = d.dot(g)

        if not math.isfinite(dg):  # numerical error -> happens if we went way too far
            dg = 1.0 # whatever >0
            alpha *= 0.1
        # if dg < dg0:
        #    print(f"Linesearch error: not convex. dg0:{dg0:.1E} dg:{dg:.1E} r:{dg/dg0:.1E} a={alpha:.1E}")
        #    print( self. getInfoStr())
        #    print("")
        #    return
        if dg > 0:
            # We went too far, have to backtrack
            self.lastNbTry = 1

            while dg > 0:
                # print( "too far, backtrack " , alpha,  alphaOptim)
                alpha *= 0.5
                x = x0 + d * alpha
                g = self.gradFun(x)
                dg = d.dot(g)
                if not math.isfinite(dg):  # numerical error
                    alpha *= 0.1
                    dg = 1.0 # whatever >0
                # self.lastNbTry +=1
        else:
            ratio = dg / dg0
            if ratio < 0.5:  # not far from overstep => decelerate a bit
                alpha *= 0.8
            else:
                alpha *= 1.2  # some margin => accelerate
        self.normgrad = np.linalg.norm(g)
        self.alpha = alpha
        return x, g

    def Search_TooComplicated(self, x0, d, g0=None):
        """
        x0: current parameter estimate. array of size n.
        d : descent direction (array of size n)
        g0: gradient at x0 (if already computed
        returning x: new parameter estimate, g: gradient at x
        """
        if g0 is None:
            g0 = self.gradFun(x0)
        dg0 = d.dot(g0)
        self.normgrad = np.linalg.norm(g0)

        if dg0 >= 0:
            print(f"Linesearch error: not a descent direction {dg0:.1E}")
            return
        alpha = self.alpha

        x = x0 + d * alpha
        g = self.gradFun(x)
        dg = d.dot(g)

        if not math.isfinite(dg):  # numerical error -> happens if we went way too far
            dg = -10 * dg0
        if dg <= dg0:
            print(
                f"Linesearch error: not convex. dg0:{dg0:.1E} dg:{dg:.1E} r:{dg/dg0:.1E}"
            )
            return
        alphaOptim = dg0 / (dg0 - dg)

        if alphaOptim < 0:
            print(f"Linesearch error: alphaOptim < 0")
            return

        if dg > 0:
            # We went too far, have to backtrack
            alpha = alphaOptim * 0.8 * 2
            self.lastNbTry = 0
            while dg > 0:
                # print( "too far, backtrack " , alpha,  alphaOptim)
                alpha = min(alpha / 2, alphaOptim * 0.8)
                x = x0 + d * alpha
                g = self.gradFun(x)
                dg = d.dot(g)
                if not math.isfinite(dg):  # numerical error
                    dg = -10 * dg0
                alphaOptim = dg0 / (dg0 - dg)
                self.lastNbTry -= 1
            self.alpha = alpha
            return x, g

        last_x_ok = x
        last_g_ok = g
        last_alpha_ok = alpha
        self.lastNbTry = 1
        while alpha < alphaOptim / 2:
            # print( "alpha < alphaOptim / 2 ", alpha ,  alphaOptim)
            self.lastNbTry += 1
            alpha = min(0.8 * alphaOptim, alpha * 4)
            x = x0 + d * alpha
            g = self.gradFun(x)
            dg = d.dot(g)
            if dg > 0 or not math.isfinite(dg):
                # print( "oups, too far" ,alpha,  "going back to" , last_alpha_ok ,   dg)
                self.alpha = last_alpha_ok
                return last_x_ok, last_g_ok
            last_x_ok = x
            last_g_ok = g
            last_alpha_ok = alpha
            alphaOptim = dg0 / (dg0 - dg)
        # print("accepting" , alpha ,  alphaOptim)
        self.alpha = alpha
        return x, g

    def getInfoStr(self):
        return (
            f"a:{self.alpha:.1E}({self.lastNbTry}),"
            + f" n:{self.nbCalls}, g:{self.normgrad:.1E}"
        )


def cos(g, g2):
    return g.dot(g2) / np.sqrt(g.dot(g) * g2.dot(g2))


class LimitedMemory:
    def __init__(self, memoryLength, n):
        self.memoryLength = memoryLength
        self.n = n
        self.data = np.zeros((memoryLength, n))
        self.nbAppend = 0

    def append(self, x):
        if self.nbAppend < self.memoryLength:
            self.data[self.nbAppend] = x
        else:
            self.data[0] = x
            self.data = np.roll(self.data, -1, axis=0)
        self.nbAppend += 1

    def Get(self):
        if self.nbAppend < self.memoryLength:
            return self.data[: self.nbAppend]
        return self.data


class LbfgsInvHessProduct:
    """
    Largely copy-pasted from:
    https://github.com/scipy/scipy/blob/master/scipy/optimize/lbfgsb.py
    Linear operator for the L-BFGS approximate inverse Hessian.
    This operator computes the product of a vector with the approximate inverse
    of the Hessian of the objective function, using the L-BFGS limited
    memory approximation to the inverse Hessian, accumulated during the
    optimization.
    Parameters
    ----------
    sk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the solution vector.
        (See [1]).
    yk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the gradient. (See [1]).
    References
    ----------
    .. [1] Nocedal, Jorge. "Updating quasi-Newton matrices with limited
       storage." Mathematics of computation 35.151 (1980): 773-782.
    """

    #  diag :  inverse diagonal hessian
    def __init__(self, n, n_corrs, diag=None):
        self.n = n
        self.n_corrs = n_corrs
        self.diag = diag
        self.sk = LimitedMemory(n_corrs, n)
        self.yk = LimitedMemory(n_corrs, n)
        self.rho = [1]

    def update(self, s, y):
        # s is the lastest update to the solution vector  ( s = dx = stepsize * d  where d is descent direction  )
        # y is the lastest update to the gradient ( y = grad(x +dx) - grad(x) )
        self.sk.append(s)
        self.yk.append(y)
        # element-wise product, then summing along axis "n_corrs" to get a vector of same len as or y
        einsum = np.einsum("ij,ij->i", self.sk.Get(), self.yk.Get())
        epsilon = 1e-15  # avoid divide by zero errors when close to optimum
        self.rho = 1 / (einsum + epsilon)

    def multiplyByInvH(self, x):
        """Efficient matrix-vector multiply with the BFGS matrices.
        This calculation is described in Section (4) of [1].
        Parameters
        ----------
        x : ndarray
            An array with shape (n,) or (n,1).
        Returns
        -------
        y : ndarray
            The matrix-vector product
        """
        s, y, rho, diag = self.sk.Get(), self.yk.Get(), self.rho, self.diag
        n_corrs = s.shape[0]
        q = np.array(x, copy=True)
        # q = np.array(x, dtype=self.dtype, copy=True)
        if q.ndim == 2 and q.shape[1] == 1:
            q = q.reshape(-1)

        alpha = np.empty(n_corrs)

        for i in range(n_corrs - 1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], q)
            q = q - alpha[i] * y[i]

        r = q
       # Multiplying by "diag" here
        if diag is not None:
            r *= diag
        for i in range(n_corrs):
            beta = rho[i] * np.dot(y[i], r)
            r = r + s[i] * (alpha[i] - beta)
        return r
