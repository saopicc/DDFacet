#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
from numpy.random import standard_normal as randn
import scipy.special

import matplotlib.pyplot as plt

from edwin import udft
from edwin.udft import urdft2 as dft
from edwin.udft import uirdft2 as idft
from edwin.udft import ir2fr
from edwin.sampling import Trace, FileTrace
import edwin.improcessing as ip

"""Module for Location Scale Mixture of Gaussian"""


def ucdeconv(data, instr_ir, user_setup=None):
    """Unsupervised convex deconvolution

    PARAMETERS
    ==========
    data: the Fourier transform of the data

    criterion: if the difference between two successive mean is less
    than this value, stop the algorithm.

    burnin: number of iteration to remove at the beginning of the
    chain to compute the mean of the image.

    max iter: maximum number of iteration.

    var_est: if set to True, compute the empirical diagonal of the
    covariance matrix of the object.  We have to return in direct
    space so this cost an FFT.
    """
    setup = {'criterion': 1e-4,
             'max iter': 200,
             'min iter': 30,
             'burnin': 30,
             'callback': None,
             'var est': True}
    if user_setup is not None:
        setup.update(user_setup)

    ## The mean of the object
    post_mean = np.zeros(data.shape)
    aux_post_mean = np.zeros(data.shape)

    post_smean = np.zeros(data.shape)
    post_var = np.zeros(data.shape)

    ## Prior law
    alpha_noise = alpha_im = alpha_aux = alpha_mean = 0
    beta_noise_bar = beta_im_bar = beta_aux_bar = beta_mean_bar = 0

    ## The instrument response in Fourier space
    instr_filter = ir2fr(instr_ir, data.shape)
    instr_gain = np.abs(instr_filter)**2

    # The correlation of the object in Fourier space
    prior_filter = udft.laplacian(2, data.shape)
    prior_gain = np.abs(prior_filter)**2

    dataf = dft(data)
    sample = np.zeros(data.shape)

    aux_sample = np.zeros_like(data)
    auxf = dft(aux_sample)
    noise_prec_chain = [1]
    gauss_prec_chain = [1]
    mean_prec_chain = [1]
    laplace_prec_chain = [1]

    delta = np.inf
    for iteration in range(setup['max iter']):
        ## Obj sample
        obj_prec = (noise_prec_chain[-1] * instr_gain +
                    gauss_prec_chain[-1] * prior_gain)
        obj_std = dft(randn(data.shape)) / np.sqrt(obj_prec)
        obj_mean = (
            noise_prec_chain[-1] * np.conj(instr_filter) * dataf +
            laplace_prec_chain[-1] * np.conj(prior_filter) * auxf) / obj_prec
        samplef = obj_std + obj_mean
        sample = idft(samplef)

        ## Aux sample
        aux_sample, _ = draw_logerf_mh(aux_sample,
                                       idft(prior_filter * samplef),
                                       gauss_prec_chain[-1],
                                       laplace_prec_chain[-1])
        auxf = dft(aux_sample)

        ## Aux prec  sample
        aux_adeq = np.sum(np.abs(aux_sample)) / 2
        laplace_prec_chain.append(npr.gamma(
            alpha_aux + aux_sample.size,
            1 / (beta_aux_bar + aux_adeq)))

        ## Noise prec sample
        data_adeq = np.sum(np.abs(dataf - instr_filter * samplef)**2) / 2
        noise_prec_chain.append(npr.gamma(
            alpha_noise + data.size / 2,
            1 / (beta_noise_bar + data_adeq)))

        ## Mean obj prec sample
        mean_prec_chain.append(npr.gamma(
            alpha_mean + 1 / 2,
            1 / (beta_mean_bar + np.sum(np.abs(sample)**2) / 2)))

        ## Obj prec sample
        prior_adeq = np.sum(np.abs(prior_filter * samplef - auxf)**2) / 2
        gauss_prec_chain.append(npr.gamma(
            alpha_im + (sample.size - 1) / 2,
            1 / (beta_im_bar + prior_adeq)))

        # Der2En0 = np.sqrt(laplace_prec_sample**2 / (8 * im_prec_chain[-1]))
        # Der2En0 = 0.5 * laplace_prec_chain[-1]**2 * (
        #     1 / (np.sqrt(pi) * Der2En0 * erfcx(Der2En0)) - 1)
        # seuil = laplace_prec_chain[-1] / Der2En0
        # mu = Der2En0 / noise_prec_chain[-1]

        ## Empirical Mean
        if iteration > (setup['burnin'] + 1):
            count = iteration - setup['burnin']
            delta = np.sum(np.abs(post_mean / (1 - count) + sample)) / (
                count * np.sum(np.abs(post_mean)))

        if iteration > setup['burnin']:
            post_mean += sample
            aux_post_mean += aux_sample

            if setup['var est']:
                post_smean += sample**2

        ## Algorithm ending by crit
        if delta < setup['criterion'] and iteration > setup['min iter']:
            break

    ## Algorithm ending by iteration
    post_mean = post_mean / (iteration - setup['burnin'])
    aux_post_mean = aux_post_mean / (iteration - setup['burnin'])
    if setup['var est']:
        post_var = post_smean / (iteration - setup['burnin']) - post_mean**2

    return (post_mean, aux_post_mean, post_var, noise_prec_chain,
            gauss_prec_chain, mean_prec_chain, laplace_prec_chain)


def supervised_cdeconv(data: np.ndarray, instr_filter: np.ndarray,
                       noise_prec: float, gauss_prec: float, laplace_prec:
                       float, setup: dict =None):
    settings = {'min iter': 50,
                'max iter': 50,
                'burnin': 20,
                'crit': 1e-4,
                'draw': False}
    settings.update({} if setup is None else setup)

    # Setup
    row_filter = udft.DiffOp(2, 0).freqr(data.shape)
    col_filter = udft.DiffOp(2, 1).freqr(data.shape)

    # Init
    data_f = dft(data)
    sample = FileTrace(burnin=settings['burnin'], init=data.copy())
    sample_f = dft(sample.last)
    auxr_sample = FileTrace(burnin=settings['burnin'],
                            init=idft(row_filter * data_f))
    auxc_sample = FileTrace(burnin=settings['burnin'],
                            init=idft(col_filter * data_f))

    var = 1 / (noise_prec * np.abs(instr_filter)**2 + gauss_prec *
               (np.abs(row_filter)**2 + np.abs(col_filter)**2))
    std = np.sqrt(var)
    Htdata = noise_prec * np.conj(instr_filter) * data_f
    Rtf = gauss_prec * np.conj(row_filter)
    Ctf = gauss_prec * np.conj(col_filter)

    for iteration in range(settings['max iter']):
        # Obj sample
        obj_mean = var * (Htdata +
                          Rtf * dft(auxr_sample.last) +
                          Ctf * dft(auxc_sample.last))
        sample_f = obj_mean + std * dft(randn(data.shape))
        sample.last = idft(sample_f)

        ## Aux sample
        auxr_sample.last, _ = draw_logerf(
            idft(row_filter * sample_f),
            laplace_prec, gauss_prec, auxr_sample.last)

        ## Aux sample
        auxc_sample.last, _ = draw_logerf(
            idft(col_filter * sample_f),
            laplace_prec, gauss_prec, auxc_sample.last)

        ## Algorithm ending by crit
        if sample.delta < settings['crit'] and iteration > settings['min iter']:
            break

        if settings['draw']:
            fig = plt.figure(1)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(sample.last)
            plt.colorbar()
            plt.gray()
            plt.title('Iteration {}'.format(iteration))
            plt.subplot(1, 3, 2)
            plt.imshow(auxr_sample.last)
            plt.colorbar()
            plt.gray()
            plt.title('Auxr')
            plt.subplot(1, 3, 3)
            plt.imshow(auxc_sample.last)
            plt.colorbar()
            plt.gray()
            plt.title('Auxc')

            plt.draw()
            fig.canvas.update()
            fig.canvas.flush_events()

    return sample, auxc_sample, auxr_sample


def semisup_cdeconv(data: np.ndarray, instr_filter: np.ndarray,
                    noise_prec: float, setup: dict =None):
    settings = {'crit': 1e-4,
                'max iter': 1000,
                'min iter': 1000,
                'burnin': 500,
                'callback': None,
                'draw': False, }
    settings.update({} if setup is None else setup)

    # Setup
    row_filter = udft.DiffOp(2, 0).freqr(data.shape)
    col_filter = udft.DiffOp(2, 1).freqr(data.shape)
    instr_gain = np.abs(instr_filter)**2
    row_gain = np.abs(row_filter)**2
    col_gain = np.abs(col_filter)**2
    alpha_prior = alpha_aux = 0
    beta_prior_bar = beta_aux_bar = 0
    data_f = dft(data)

    # Init
    sample = FileTrace(burnin=settings['burnin'], init=data.copy())
    sample_f = dft(sample.last)
    auxr_sample = FileTrace(burnin=settings['burnin'],
                            init=idft(row_filter * data_f))
    auxc_sample = FileTrace(burnin=settings['burnin'],
                            init=idft(col_filter * data_f))
    auxr_f = dft(auxr_sample.last)
    auxc_f = dft(auxc_sample.last)
    gauss_prec = Trace(burnin=settings['burnin'], init=1)
    laplace_prec = Trace(burnin=settings['burnin'], init=1)

    for iteration in range(settings['max iter']):
        # Obj sample
        obj_prec = (noise_prec * instr_gain +
                    gauss_prec.last * row_gain +
                    gauss_prec.last * col_gain)
        obj_mean = (
            noise_prec * np.conj(instr_filter) * data_f +
            gauss_prec.last * (
                np.conj(row_filter) * auxr_f +
                np.conj(col_filter) * auxc_f)) / obj_prec
        obj_std = dft(randn(data.shape)) / np.sqrt(obj_prec)
        sample_f = obj_mean + obj_std
        sample.last = idft(sample_f)

        # Aux sample
        auxr_sample.last, _ = draw_logerf(
            idft(row_filter * sample_f),
            laplace_prec.last, gauss_prec.last,
            auxr_sample.last)
        auxr_f = dft(auxr_sample.last)

        # Aux sample
        auxc_sample.last, _ = draw_logerf(
            idft(col_filter * sample_f),
            gauss_prec.last, laplace_prec.last,
            auxc_sample.last)
        auxc_f = dft(auxc_sample.last)

        # Prior prec sample
        prior_adeq = (np.sum(abs(col_filter * sample_f - auxc_f)**2) +
                      np.sum(abs(row_filter * sample_f - auxr_f)**2)) / 2
        # gauss_prec.last = npr.gamma(alpha_prior + (sample.size - 1) / 2,
        #                             1 / (beta_prior_bar + prior_adeq))
        # Over relaxation
        tested = np.sort(np.append(
            npr.gamma(alpha_prior + (sample.size - 1) / 2,
                      1 / (beta_prior_bar + prior_adeq),
                      size=15),
            gauss_prec.last))
        idx = np.nonzero(tested == gauss_prec.last)[0][0]
        gauss_prec.last = tested[len(tested) - idx - 1]

        # Aux prec sample
        aux_adeq = (np.sum(np.abs(auxc_sample.last)) +
                    np.sum(np.abs(auxr_sample.last))) / 2
        laplace_prec.last = npr.gamma(alpha_aux + 2 * sample.size,
                                  1 / (beta_aux_bar + aux_adeq))

        ## Algorithm ending by crit
        if sample.delta < settings['crit'] and iteration > settings['min iter']:
            break

        if settings['draw']:
            fig = plt.figure(1)
            plt.clf()
            plt.subplot(1, 3, 1)
            plt.imshow(sample.last)
            plt.colorbar()
            plt.gray()
            plt.title('Iteration {}'.format(iteration))
            plt.subplot(1, 3, 2)
            plt.imshow(auxr_sample.last)
            plt.colorbar()
            plt.gray()
            plt.title('Auxr')
            plt.subplot(1, 3, 3)
            plt.imshow(auxc_sample.last)
            plt.colorbar()
            plt.gray()
            plt.title('Auxc')

            plt.draw()
            fig.canvas.update()
            fig.canvas.flush_events()

    return sample, auxc_sample, auxr_sample, gauss_prec, laplace_prec


def draw_logerf(b_bar, gauss_prec, laplace_prec, previous=None):
    """
    Notes de Gio
    ============

    Il y en a au moins deux version d'un point de vue numerique

    - Une version tout a la ERFC (ie a la Chi)

    - Une version a la ERFCX (ie a la Psi)

    Ici c'est la version a la Erfc, c'est a dire a la Chi.
    """
    rho = laplace_prec / (2 * gauss_prec)

    theta_m = scipy.special.erfc(np.sqrt(gauss_prec / 2) * (rho + b_bar))
    theta_m *= np.exp(+laplace_prec * b_bar / 2)
    theta_p = scipy.special.erfc(np.sqrt(gauss_prec / 2) * (rho - b_bar))
    theta_p *= np.exp(-laplace_prec * b_bar / 2)
    theta = theta_m + theta_p

    # Uniform simulation and linear transformation
    rand_u = npr.uniform(size=b_bar.shape)

    # Two cases
    idx_m = rand_u < (theta_m / theta)
    idx_p = rand_u >= (theta_m / theta)

    # CDF inversion
    sample = np.full_like(b_bar, np.nan)
    sample[idx_m] = b_bar[idx_m] + rho - np.sqrt(2 / gauss_prec) * \
        scipy.special.erfinv(
            scipy.special.erf(np.sqrt(gauss_prec / 2) *
                              (b_bar[idx_m] + rho)) -
            (theta * rand_u - theta_m)[idx_m] *
            np.exp(-laplace_prec * b_bar[idx_m] / 2))
    sample[idx_p] = b_bar[idx_p] - rho - np.sqrt(2 / gauss_prec) * \
        scipy.special.erfinv(
            scipy.special.erf(np.sqrt(gauss_prec / 2) *
                              (b_bar[idx_p] - rho)) -
            (theta * rand_u - theta_m)[idx_p] *
            np.exp(+laplace_prec * b_bar[idx_p] / 2))

    # Profile version
    # sample_m = scipy.special.erf(np.sqrt(gauss_prec / 2) * (b_bar[idx_m] + rho))
    # sample_m -= (theta * rand_u - theta_m)[idx_m] * np.exp(-laplace_prec * b_bar[idx_m] / 2)
    # sample_m = scipy.special.erfinv(sample_m)
    # sample[idx_m] = b_bar[idx_m] + rho - np.sqrt(2 / gauss_prec) * sample_m

    # sample_p = scipy.special.erf(np.sqrt(gauss_prec / 2) * (b_bar[idx_p] - rho))
    # sample_p -= (theta * rand_u - theta_m)[idx_p] * np.exp(+laplace_prec * b_bar[idx_p] / 2)
    # sample_p = scipy.special.erfinv(sample_p)
    # sample[idx_p] = b_bar[idx_p] - rho - np.sqrt(2 / gauss_prec) * sample_p

    accept = np.ones_like(sample)
    if previous is not None:
        idx = ~np.isfinite(sample)
        sample_mh, accept_mh = draw_logerf_mh(previous[idx], b_bar[idx], gauss_prec, laplace_prec)
        sample[idx] = sample_mh
        accept[idx] = accept_mh
    else:
        assert np.all(np.isfinite(theta_m)), 'Nan/Inf dans theta_m'
        assert np.all(np.isfinite(theta_p)), 'Nan/Inf dans theta_p'
        assert np.all(theta != 0), 'Des 0 dans theta'
        assert np.all(np.isfinite(sample)), 'NaN/Inf dans sample'

    return sample, accept


def draw_logerf_mh(previous, b_bar, gauss_prec, laplace_prec):
    coeff = 1 / np.sqrt(2 * np.log(2))

    rho = laplace_prec / (2 * gauss_prec)

    # loi de proposition sous une loi gaussienne de moyenne le max de
    # la proba conditionnel d'ecart type proto la largeur a mi-hauteur

    # Calcul de la moyenne de la loi

    idx_m = b_bar < -rho
    idx_cm = (-rho <= b_bar) * (b_bar < 0)
    idx_cp = (0 <= b_bar) * (b_bar <= rho)
    idx_p = rho < b_bar
    idx_c = idx_cp + idx_cm

    prop_mean = np.full_like(b_bar, np.nan)
    prop_mean[idx_p] = b_bar[idx_p] - rho
    prop_mean[idx_m] = b_bar[idx_m] + rho
    prop_mean[idx_c] = 0

    prop_std = np.full_like(b_bar, np.nan)
    prop_std[idx_p] = np.sqrt(1 / gauss_prec)
    prop_std[idx_m] = np.sqrt(1 / gauss_prec)

    prop_std[idx_cp] = coeff * (b_bar[idx_cp] - rho + np.sqrt((b_bar[idx_cp] - rho)**2 + 1 / gauss_prec))

    prop_std[idx_cm] = -coeff * (b_bar[idx_cm] + rho - np.sqrt((b_bar[idx_cm] + rho)**2 + 1 / gauss_prec))

    proposed = npr.standard_normal(b_bar.shape) * prop_std + prop_mean

    # Rapport
    criterion = (gauss_prec / 2 * (np.abs(b_bar - previous)**2 -
                                   np.abs(b_bar - proposed)**2) +
                 laplace_prec / 2 * (np.abs(previous) - np.abs(proposed)) +
                 ((proposed - prop_mean)**2) / (2 * prop_std**2) -
                 ((previous - prop_mean)**2) / (2 * prop_std**2))

    realized = np.log(npr.uniform(size=b_bar.shape))

    return (np.where(realized < criterion, proposed, previous),
            np.where(realized < criterion, 1, 0))


def draw_gauss_laplace(gauss_mean: np.array,
                       gauss_prec: float,
                       laplace_prec: float,
                       max_reject: int=10):
    """Simulate gauss-laplace law by reject method

    p(x) ∝ exp(- γ₁|x - m|²/2 - γ₂|x|/2 )

    """
    rho = laplace_prec / (2 * gauss_prec)
    maximum = np.where(np.abs(gauss_mean) > rho,
                       gauss_mean - np.sign(gauss_mean) * rho,
                       0)

    still_rejected = np.full_like(gauss_mean, True, dtype=bool)
    sample = np.full_like(gauss_mean, np.nan)

    def log_crit(prop, maximum, gauss_mean):
        return (gauss_prec * (maximum**2 -
                              maximum * (gauss_mean + prop) +
                              gauss_mean * prop) +
                2 * laplace_prec * (np.abs(maximum) - np.abs(prop)))

    while (np.any(still_rejected) and
           next(iter(range(max_reject)), max_reject) < max_reject):
        still_rejected_max = maximum[still_rejected]
        proposition = still_rejected_max + randn(
            (np.count_nonzero(still_rejected),)) / np.sqrt(gauss_prec)

        crit = log_crit(proposition,
                        still_rejected_max,
                        gauss_mean[still_rejected])
        accepted = np.log(npr.uniform(size=proposition.size)) < crit

        sample[still_rejected] = np.where(accepted, proposition, np.nan)
        still_rejected[still_rejected] = np.where(accepted, False, True)
    else:
        return sample

    raise ValueError('Maximum number of rejection reached')
