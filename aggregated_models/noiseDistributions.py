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



def vectorizedLineSearch( f , mins, maxs, nbiters = 100 ):
    
    mins = mins * 1.0
    maxs = maxs * 1.0
    
    fmins = f(mins)
    fmaxs = f(maxs)
    
    if any ( fmins*fmaxs > 0) : 
        raise( "wrong signs" )
    
    for i in range( 0, nbiters ):
        means = (mins+maxs) / 2.0
        fmeans = f(means)

        #print(i,  mins, maxs, fmins, fmeans,  fmaxs )
        
        index_replace_min = np.where (fmins * fmeans > 0 )
        index_replace_max = np.where (fmins * fmeans <= 0 )
        mins[ index_replace_min ] = means[index_replace_min]
        maxs[ index_replace_max ] = means[index_replace_max]
        fmins[ index_replace_min ] = fmeans[index_replace_min]
        fmaxs[ index_replace_max ] = fmeans[index_replace_max]


    means = (mins+maxs) / 2
    return means    


def expectedGaussianKnowingDataPlusNoiseAndSampledDataExpect(noisyObservedCount, 
                                                             scaledPrediction, 
                                                             sigma, 
                                                             N,  # number of samples
                                                             M,  # number of Gibbs samples
                                                             nbiters = 50,
                                                         minvalue=0.01):
    # approximates E(G | G+A = noisyObservedCount ;
    #                    G~Gaussian(sigma) ; 
    #                    A~Binomial( lambda , N ) ; 
    #                    lambda ~ beta(  p , M-p  )
    #              with p :=  scaledPrediction * M / N
    # by solving:  g+ sigma² log( 1 + p /(d-g)) = sigma² log( 1 + M /( N -d+g ) )
    #
    # Why?
    # we first approximate the expectation by a mode: 
    #  E(G | G+A = noisyObservedCount )~= Argmax_g P(G =g | G+A = noisyObservedCount )
    # and we can write:
    #  Argmax_g P(G =g | G+A = noisyObservedCount ) propto P( G=g ) P( A = noisyObservedCount - g)  
    #  We need a formula for P( A = noisyObservedCount - g):
    #  - A follows a binomial bistribution, of parameters n  and lambda , with: 
    #       - n the number of samples in the agg data
    #       - lambda the propability of this projection according to the model
    #  so P(A=a  | lambda) =  ... binomial formula 
    #  problem: we do not know lambda exactly! we only have an estimate built from M Gibbs samples of the model
    #  more precisely: we have M Gibbs samples, and among them p = scaledPrediction * M *1.0 / N are positive
    #  We thus may use a beta prior on lambda : beta(p , M-p) 
    #
    # we now need to solve:  Armax_g P(G=g) Sum_{lambda in [0,1]}   P(A=a  | lambda) . beta(p , M-p)
    # the integral is a the form : int_[0,1] x^a (1-x)^b dx 
    #  we use int_[0,1] x^a (1-x)^b dx  = ( integration par parties + recurence ) = a!b!/( a+b+ )!
    #
    #  using log(k!) ~= n^n e-n  , writing that dP/dg = 0, and doing the caclculus, we find the formula: 
    #  g+ sigma² log( 1 + p /(d-g)) = sigma² log( 1 + M /( N -d+g ) ) = 0
    
    from scipy import optimize
    
    d = noisyObservedCount
    scaledPrediction[scaledPrediction<=0 ] = 0.01
    p = scaledPrediction * M *1.0 / N
    s2 = sigma * sigma
        
    def f(x):
        return x + s2* np.log( 1 + p /(d-x)) - s2 *  np.log( 1 + (M - p) /( N -d+x ) ) 
    eps = 0.000001
    
    mins = d-N +eps
    maxs = d - eps

    #print( f(mins) )
    #print( f(maxs) )
    
    myroot = vectorizedLineSearch( f , mins, maxs, nbiters = nbiters )
    return myroot

    
