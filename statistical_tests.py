import scipy.stats
import numpy as np


def stat_test_on_theta(periodic_data, sample_size):
    KS_test_result = []
    MW_test_result = []
    for m in range(100):
        # scotts factor
        np.random.seed(m)
        uniform = np.random.uniform(
            low=np.min(periodic_data),
            high=np.max(periodic_data),
            size=periodic_data.size,
        )
        sample1 = np.random.choice(uniform, size=sample_size, replace=True, p=None)
        periodic_sample = np.random.choice(
            np.ravel(periodic_data), size=sample_size, replace=True, p=None
        )
        print(f"Uniform vs. My data: {scipy.stats.ks_2samp( periodic_sample,sample1)}")
        KS_test_result.append(scipy.stats.ks_2samp(periodic_sample, sample1)[1])
    # MW_test_result.append(scipy.stats.mannwhitneyu(  periodic_sample,sample1)[1])

    return KS_test_result, MW_test_result


# sns.set_theme(font_scale=1.5, rc={'text.usetex' : True})


def producing_random_points_with_theta(number_of_points, rand_int):
    rng = np.random.default_rng(rand_int)
    Phi = np.arccos(1 - 2 * (rng.random((number_of_points))))

    Theta = 2 * np.pi * rng.random((number_of_points))
    rho = 1  # 7.7942286341
    A = Phi
    B = Theta
    R = np.array(
        [rho * np.sin(A) * np.cos(B), rho * np.sin(B) * np.sin(A), rho * np.cos(A)]
    )

    return Phi, Theta, R

    # scotts factor


def stat_test_on_phi(periodic_data, sample_size):
    KS_test_result = []
    MW_test_result = []
    for m in range(100):
        Phi, Theta, R = producing_random_points_with_theta(periodic_data.size, m)

        sample_sin = np.random.choice(Phi, size=sample_size, replace=True, p=None)
        periodic_sample = np.random.choice(
            np.ravel(periodic_data), size=sample_size, replace=True, p=None
        )
        KS_test_result.append(scipy.stats.ks_2samp(periodic_sample, sample_sin)[1])
        # MW_test_result.append(scipy.stats.mannwhitneyu( periodic_sample,sample_sin)[1])

        print(
            f"sampled sine vs. My data sample KS test: {scipy.stats.ks_2samp( periodic_sample,sample_sin)}"
        )
        # MW only makes sense in ordinal data - no natural ranking
    # print(f'sampled sine vs. My data sample Mannwhitney test: {scipy.stats.mannwhitneyu( periodic_sample,sample_sin)}')
    # print(f'sampled sine vs. My data sample ranksums test: {scipy.stats.ranksums( periodic_sample,sample_sin)}')

    return KS_test_result, MW_test_result


def generic_stat_kolmogorov_2samp(dist1, dist2):
    KS_test_result = scipy.stats.ks_2samp(dist1, dist2)

    return KS_test_result
