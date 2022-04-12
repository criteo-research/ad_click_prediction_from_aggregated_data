import numpy as np


class LaplaceNoise:  # Distrete Laplace
    def __init__(self, epsilon):
        self.epsilon = epsilon
        decay = np.exp(-epsilon)
        self.normalization = (1 - decay) / (1 + decay)
        self.sigma = np.sqrt(2 * decay) / (1 - decay)

    def Proba(self, x):
        return self.normalization * np.exp(-self.epsilon * np.abs(x))

    def LogProba(self, x):  # avoiding numerical issues with calling np.log(self.proba(x)) for large x
        return -self.epsilon * np.abs(x) + np.log(self.normalization)

    def Sample(self, n):
        decay = np.exp(-self.epsilon)
        p0 = (1 - decay) / (1 + decay)
        return (
            np.random.geometric(1 - decay, n) * (np.random.randint(0, 2, n) * 2 - 1) * (np.random.uniform(0, 1, n) > p0)
        )

    def expectedNoiseApprox(self, observedCount, poisson):
        return expectedLaplaceAffineApprox(observedCount, poisson, self.epsilon)


class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma
        self.normalization = 1 / (self.sigma * np.sqrt(2 * np.pi))

    def Proba(self, x):
        return self.normalization * np.exp(-x * x / (2 * self.sigma * self.sigma))

    def LogProba(self, x):
        return -x * x / (2 * self.sigma * self.sigma) + np.log(self.normalization)

    def Sample(self, n):
        return np.random.normal(0, self.sigma, n)

    def expectedNoiseApprox(self, observedCount, poisson):
        return expectedGaussianApprox(observedCount, poisson, self.sigma)


def PoissonProba(n, c0):
    if n < 0:
        return 0.0
    return np.exp(n * np.log(c0) - c0 - np.math.lgamma(n + 1))


def expectedNoise_ByIntegration(observedCount, poisson, noiseDistribution, maxnoise=1000):
    # estimates E(N | N+P = observedCount ; N~noiseDistribution ; P~Poisson(poisson)  ) by numeric integration
    poisson_min = max(0, int(observedCount - maxnoise))
    poisson_max = int(observedCount) + maxnoise
    poisson_range = np.arange(poisson_min, poisson_max - poisson_min)

    a = sum(
        [
            (observedCount - k) * noiseDistribution.Proba(observedCount - k) * PoissonProba(k, poisson)
            for k in poisson_range
        ]
    )
    b = sum([noiseDistribution.Proba(observedCount - k) * PoissonProba(k, poisson) for k in poisson_range])
    return a / b


def expectedGaussianApprox(observedCount, poissonExpect, sigma, 
                           scaleFactor = 1.0,
                           nbiters=10, minvalue=0.01):
    # approximates E(G | G+P = observedCount ; G~Gaussian(sigma) ; P~Poisson(poisson)  ) by numeric integration
    # by solving:    n+sigma²ln(n) = observed +sigma²ln(poissonExpect)
    # indeed:
    #  - the mode of the posterior Poisson verify:  n+sigma²ln(n) = observed +sigma²ln(poissonExpect) - sigma²/2n
    #  - maximum of n * P(N=n|P+G=observed) verify: n+sigma²ln(n) = observed +sigma²ln(poissonExpect) + sigma²/2n
    #  - expectation should be around those two values, and so should be the solution of the proposed equation

    # in practical cases poissonExpect is actaully scaled (because it is computed from less samples than the obeervedCount) . We remove this scaling to get a true Poisson distrib, and thus also scale the observed count and sigma of gaussian noise
    sigma = sigma / scaleFactor
    observedCount = observedCount/scaleFactor
    poissonExpect = poissonExpect/scaleFactor    
    
    s2 = sigma * sigma
    y = observedCount + s2 * np.log(poissonExpect + 0.001)

    n = y  # initial guess
    n = n * (n > 0) + minvalue  # avoiding log(negative) or log(0)

    for i in range(0, nbiters):
        f = n + s2 * np.log(n) - y
        df = 1 + s2 / n
        dn = -f / df
        n = n + dn
        n = n * (n > 0) + minvalue

    g = observedCount - n
    return g
# typo kept for retrocompatibility
expectedGaussiaApprox = expectedGaussianApprox


def expectedLaplaceAffineApprox(observedCount, poisson, epsilon):
    # approximates E(L | L+P = observedCount ; L~Laplace(espilon) ; P~Poisson(poisson)  )
    # by a piecewise affine function of observedCount with 3 pieces.

    expEps = np.exp(epsilon)
    lowerCut = poisson / expEps
    upperCut = poisson * expEps

    p1 = 0.95  # slope between the two cut points
    p2 = 0.05  # slope before and after the two cut points

    # note:  looking at the curves, the true values seems actually to be 1.0 and 0.0.
    # Avoiding the 0.0 however might avoid getting a zero gradient in this area, and might avoid complications.

    valAtLowerCut = poisson + p1 * (lowerCut - poisson)
    valAtUpperCut = poisson + p1 * (upperCut - poisson)

    expectedPoisson = (
        (observedCount > lowerCut) * (observedCount < upperCut) * (poisson + p1 * (observedCount - poisson))
    )
    expectedPoisson += (observedCount <= lowerCut) * (valAtLowerCut + p2 * (observedCount - lowerCut))
    expectedPoisson += (observedCount >= upperCut) * (valAtUpperCut + p2 * (observedCount - upperCut))
    return observedCount - expectedPoisson










def expectedGaussianKnowingDataPlusNoiseAndSampledDataExpect(observedCount, 
                                                         scaledPoissonExpect, 
                                                         sigma, 
                                                         scaleFactor = 1.0,
                                                         nbiters=10, 
                                                         minvalue=0.01):
    # approximates E(G | G+P = observedCount ;
    #                    G~Gaussian(sigma) ; 
    #                    P~Poisson( lambda N ) ; 
    #                    M.lambda ~ Gamma(  poissonExpect ) , 1 )  )
    #              with scaleFactor := N/M
    #              and scaledPoissonExpect = poissonExpect * scaleFactor
    # 
    # by solving:    n+sigma²ln( n / (n+p-1)) = observed - sigma²ln( (N+M)/N )     
    poissonExpect = scaledPoissonExpect / scaleFactor -1
    poissonExpect[ poissonExpect < 0 ] = 0
    
    s2 = sigma * sigma
    y = observedCount - s2 * np.log( 1 + 1 / scaleFactor )

    n = y  # initial guess
    n = n * (n > 0) + minvalue  # avoiding log(negative) or log(0)

    for i in range(0, nbiters):
        #f = n + s2 * np.log(n/( n + poissonExpect ) ) - y
        #df = 1 + s2 / n
        #dn = -f / df
        #n = n + dn
        n = y + s2 * np.log( ( n + poissonExpect ) /n )
        n = n * (n > 0) + minvalue

    g = observedCount - n
    return g

