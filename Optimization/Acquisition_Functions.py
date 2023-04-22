import numpy as np
import numpy.typing as npt
from scipy.stats import norm

def probability_improvement(X, X_Sample, gpr, xi=0.01):
    """
    Probability Improvement Acquisition function
    Compute probability of improvement at X using Sample points X_sample

    Args:
        X: Point to compute PI
        X_Sample: Array of sampled locations
        gpr: Gaussian Process Regressor
        xi: Trade-off Param(From PI definition)
    """
    stdX = np.std(X)
    if stdX == 0:
        return 0 #if stdX = 0 no variance means no probability for improvement so return 0
    prediction=gpr.predict(X_Sample)
    Z = (np.mean(X, axis = 1) - np.mean(prediction) - xi) / stdX
    return norm.cdf(Z)

def expected_improvement(X, X_Sample, gpr, xi=0.01):
    """
    Expected Improvement Acquisition improvement
    Instead of probability of improvement compute the expected improvement

    Args: Same as above
    """
    stdX = np.std(X)
    if stdX == 0:
        return 0
    m_gp = gpr.predict(X_Sample)
    Zi = (np.mean(X, axis=1) - np.mean(m_gp) -xi)
    Z = Zi/stdX
    return (Zi*norm.cdf(Z) + stdX*norm.pdf(Z))
