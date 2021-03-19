import numpy as np
from agg_models import noiseDistributions


class GaussianMechanism:
    def __init__(self, epsilon, delta, nbquerries):
        self.epsilon = epsilon
        self.delta = delta
        self.nbquerries = nbquerries
        self.l2sensitivity = np.sqrt(nbquerries)
        self.sigma = GaussianMechanism.GaussianMechanism_2(self.epsilon, self.delta, self.l2sensitivity)

    def getNoise(self):
        return noiseDistributions.GaussianNoise(self.sigma)

    def __repr__(self):
        return f"GaussianMechanism epsilon:{self.epsilon} delta:{self.delta} sigma:{self.sigma}"

    @staticmethod
    def GaussianMechanism_2(epsilon, delta, l2sensitivity):
        # https://arxiv.org/pdf/1911.12060.pdf
        # section 5
        c2 = np.log(2 / (np.sqrt(16 * delta + 1) - 1))
        c = np.sqrt(c2)
        sigma = l2sensitivity * (c + np.sqrt(c2 + epsilon)) / (epsilon * np.sqrt(2))
        return sigma

    @staticmethod
    def GaussianMechanism_Dwork2014(epsilon, delta, l2sensitivity):
        return np.sqrt(2 * np.log(1.25 / delta)) * l2sensitivity / epsilon


class LaplaceMechanism:
    def __init__(self, epsilon, nbquerries):
        self.epsilon = epsilon
        self.nbquerries = nbquerries
        self.laplaceScale = epsilon / nbquerries

    def getNoise(self):
        return noiseDistributions.LaplaceNoise(self.laplaceScale)

    def __repr__(self):
        return f"LaplaceMechanism epsilon:{self.epsilon}  scale:{self.laplaceScale} sigma:{self.getNoise().sigma}"
