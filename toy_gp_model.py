
import numpy as np
from scipy import (linalg, stats)

from sami import DataCube

import matplotlib.pyplot as plt

# Let's model emission line amplitude as a GP in radius and theta
import george
import george.kernels

np.random.seed(0)

# Generate a random covariance matrix.
# (A valid covariance matrix must be positive semi-definite).
true_metric = np.diag(np.random.uniform(size=2))
#true_metric[0, 1] = true_metric[1, 0] = np.random.uniform(-0.01, 0.01) * np.prod(np.sqrt(np.diag(true_metric)))

true_mean = 10
true_period = 2 * np.pi
true_gamma = np.random.uniform()
true_scale = np.random.uniform(0, 10)

radius_kernel = george.kernels.ExpSquaredKernel(
    metric=true_metric, 
    ndim=2,
    axes=[0, 1]
)
theta_kernel = george.kernels.ExpSine2Kernel(
    gamma=true_gamma,
    log_period=np.log(true_period),
    ndim=2,
    axes=1
)

kernel = true_scale * radius_kernel * theta_kernel

gp = george.GP(
    kernel, 
    mean=true_mean
)

# Generate a grid 
x_px_lims = y_px_lims = (-1, 1)
N_x_px, N_y_px = (25, 25)
x = np.linspace(*x_px_lims, N_x_px)
y = np.linspace(*y_px_lims, N_y_px)
xy_mesh = np.meshgrid(x, y)
X = np.array(xy_mesh).reshape((2, -1)).T

# convert to r and theta.
def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return np.array([r, theta]).T

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y]).T


X_polar = cartesian_to_polar(*X.T)

# We want the GP to be in polar coordinates.
gp.compute(X_polar)

# Draw a prior from the GP.
z = gp.sample(size=1)

# OK, now z is going to be the prediction for emission line amplitude.

# Mask out regions further away than some arbitrary radius.
masked = (X_polar[:, 0] > 0.8)
z[masked] = np.nan

# Plot the prior sample from the GP
shape = (N_x_px, N_y_px)
fig, ax = plt.subplots()
ax.contourf(
    x, y,
    z.reshape((N_x_px, N_y_px))
)

# Let's generate the fake galaxy data.

# Load a data cube to get dimensions, dispersion, etc.
cube = DataCube("273952_cube_red.fits.gz")
xc, yc = (np.array(cube.flux.shape[1:])/2).astype(int)
dispersion = cube.dispersion
central_spaxal_flux = cube.flux[:, xc, yc].astype('float64')

fig, ax = plt.subplots()
ax.plot(dispersion, central_spaxal_flux)

# Let's just get the median and standard deviation of flux, and assume
# the flux will be roughly constant across spatial pixels.
f = central_spaxal_flux[dispersion > 7000]
flux_mu, flux_std = (np.nanmean(f), np.nanstd(f))
# Note: flux_mu \approx 0.02, flux_std \approx 0.002

# Now let's assume there is an emission line at 6739 angstroms,
# with the same standard deviation everywhere, but the amplitude
# is set by the gaussian process.
true_wavelength = 6739.0
galaxy_dispersion = np.tile(dispersion, xc * xc).reshape((xc, yc, -1))

# In a more complex model we would be modelling the location of wavelengths nad 
wavelength = true_wavelength * np.ones((xc, yc))
sigma = 0.5 * np.ones((xc, yc))

amplitude = z.reshape((xc, yc))

relative_emission_flux = amplitude[:, :, None] \
                       * np.exp(-0.5 * ((wavelength[:, :, None] - galaxy_dispersion)/sigma[:, :, None])**2)
relative_emission_flux += 1

galaxy_flux = flux_mu * np.ones((xc, yc, dispersion.size))
galaxy_flux *= relative_emission_flux

# Assume white noise everywhere.
galaxy_flux += np.random.normal(0, flux_std, size=(xc, yc, dispersion.size))
galaxy_ivar = np.random.normal(0, flux_std, size=(xc, yc, dispersion.size))**-2

# Plot the 'galaxy spectra' for two random spaxels to check the amplitudes
# are actually different.
fig, ax = plt.subplots()
ax.plot(dispersion, galaxy_flux[15, 20])
ax.plot(dispersion, galaxy_flux[17, 13])

# Now let's try to infer the emission width.
# Method 1: Fit the emission line for each spaxel and record it's amplitude.
# Method 2: Fit all spaxels and all pixels simultaneously using a GP.
# Then compare the error on both methods to the true model.


import pymc3 as pm

# Method 2.
with pm.Model() as latent_gp_model:

    input_dim = 2 # number of latent dimensions for GP (radius, theta)

    # TODO: Find the lengthscale, don't fit it.
    rk = pm.gp.cov.ExpQuad(
        input_dim=input_dim, 
        ls=np.diag(true_metric), 
        active_dims=[0, 1]
    )
    """
    Fuck PyMC and their stupid non-standard definitions of kernel functions.
    https://docs.pymc.io/api/gp/cov.html#pymc3.gp.cov.Periodic
    
    Periodic in PyMC3:

    k(x, x') = \exp{-\frac{\sin^2{\pi|x-x'|\frac{1}{T}}}{2l^2}}

    ExpSine2Kernel in George:

    k(x, x') = \exp{-\Gamma\sin^2{\frac{\pi}{P}|x - x'|}}

    So:
        \Gamma = \frac{1}{2l^2}
    Or:
        l = \sqrt{\frac{1}{2\Gamma}}
    """
    # TODO: Find the periodic lengthscale, don't fit it.
    true_tk_ls = np.sqrt(1/(2 * true_gamma))
    tk = pm.gp.cov.Periodic(
        input_dim=input_dim,
        period=true_period,
        ls=true_tk_ls,
        active_dims=[1],
    )

    # Build the mean and covariance functions.
    # TODO: Don't use the true mean value. Fit it instead.
    mean_func = pm.gp.mean.Constant(c=true_mean)
    # TODO: Don't use the true scale value. Fit it instead.
    cov_func = true_scale * tk * rk

    gp = pm.gp.Latent(
        mean_func=mean_func,
        cov_func=cov_func
    )

    # We need to define the number of points we can afford to have
    # in latent space.
    N_latent_per_dim = (5, 5)
    X_latent = np.array(
        np.meshgrid(
            np.linspace(0, 1, N_latent_per_dim[0]), # radius
            np.linspace(0, true_period, N_latent_per_dim[1]) # theta
        )
    ).reshape((input_dim, -1)).T

    # Place a GP prior over the true emission amplitude.
    omega = gp.prior(
        f"amplitude_{true_wavelength:.0f}", 
        X=X_latent
    )

    '''
    l1_ = pm.Normal(
        'l1',
        mu=pm_splittings(omega, 1),
        sd=e_l1_splittings,
        observed=l1_splittings
    )

    l2_ = pm.Normal(
        'l2',
        mu=pm_splittings(omega, 2),
        sd=e_l2_splittings,
        observed=l2_splittings
    )

    l3_ = pm.Normal(
        'l3',
        mu=pm_splittings(omega, 3),
        sd=e_l3_splittings,
        observed=l3_splittings
    )

    trace = pm.sample(10000, tune=5000, chains=1)
    '''

raise a



dimension_names = ("radius", "theta", "dispersion")
ndim = len(dimension_names)

kernel = 1 * george.kernels.ExpSquaredKernel(metric=np.eye(2), ndim=ndim, axes=[0, 1]) \
           * george.kernels.ExpSine2Kernel(gamma=1, log_period=np.log(2 * np.pi), ndim=ndim, axes=1)
           
gp = george.GP(
    kernel, 
    mean=0.025, fit_mean=True, 
    white_noise=np.log(0.0025), fit_white_noise=True
)

# Freeze the period parameter.
gp.freeze_parameter("kernel:k2:log_period")

# First we need polar coordinates.

raise a

def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(Y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(Y, quiet=True)

