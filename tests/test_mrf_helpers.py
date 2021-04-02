import bisect
import numpy as np
from aggregated_models.mrf_helpers import mybisect


# Checking that mybisect indeed returns the same value as bisect.bisect:
def test_mybisect():
    a = np.array([1.0, 2.0, 3.0])
    testedvalues = list(a) + list(np.random.random(100) * 4)
    for x in testedvalues:
        assert bisect.bisect(a, x) == mybisect(a, x)
