
import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """

    # Calculate PMF: Probability of exactly k successes
    pmf = (
        math.comb(n, k) *
        (p ** k) *
        ((1 - p) ** (n - k))
    )

    # Calculate CDF: Probability of at most k successes
    cdf = 0.0

    for i in range(k + 1):
        probability = (
            math.comb(n, i) *
            (p ** i) *
            ((1 - p) ** (n - i))
        )
        cdf += probability

    # Return results
    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }


# Example
print(binomial_pmf_cdf(5, 0.5, 2))
print(binomial_pmf_cdf(10, 0.3, 0))
print(binomial_pmf_cdf(8, 0.7, 8))
