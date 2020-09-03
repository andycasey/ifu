
import numpy as np
from scipy import (interpolate, linalg, stats)

import theano.tensor as tt

from sami import DataCube

import matplotlib.pyplot as plt

# Let's model emission line amplitude as a GP in radius and theta
import george
import george.kernels

random_seed = 0
np.random.seed(random_seed)

# Generate a random covariance matrix.
# (A valid covariance matrix must be positive semi-definite).
'''
L = np.random.randn(ndim, ndim)
L[np.diag_indices_from(L)] = 0.1*np.exp(L[np.diag_indices_from(L)])
L[np.triu_indices_from(L, 1)] = 0.0
cov = np.dot(L, L.T)
'''
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
flux_mu, flux_std = (np.nanmean(f), 10 * np.nanstd(f))
# Note: flux_mu \approx 0.02, flux_std \approx 0.002

# Now let's assume there is an emission line at 6739 angstroms,
# with the same standard deviation everywhere, but the amplitude
# is set by the gaussian process.
true_wavelength = 6739.0
true_sigma = 0.5
galaxy_dispersion = np.tile(dispersion, xc * xc).reshape((xc, yc, -1))

# In a more complex model we would be modelling the location of wavelengths nad 
wavelength = true_wavelength * np.ones((xc, yc))
sigma = true_sigma * np.ones((xc, yc))

amplitude = z.reshape((xc, yc))

relative_emission_flux = amplitude[:, :, None] \
                       * np.exp(-0.5 * ((wavelength[:, :, None] - galaxy_dispersion)/sigma[:, :, None])**2)
relative_emission_flux += 1

galaxy_flux = flux_mu * np.ones((xc, yc, dispersion.size))
galaxy_flux *= relative_emission_flux

# Assume white noise everywhere.
galaxy_flux += np.random.normal(0, flux_std, size=(xc, yc, dispersion.size))
galaxy_err = np.abs(np.random.normal(0, flux_std, size=(xc, yc, dispersion.size)))

# Plot the 'galaxy spectra' for two random spaxels to check the amplitudes
# are actually different.
fig, ax = plt.subplots()
ax.plot(dispersion, galaxy_flux[15, 20])
ax.plot(dispersion, galaxy_flux[17, 13])

# Now let's try to infer the emission width.
# Method 1: Fit the emission line for each spaxel and record it's amplitude.
# Method 2: Fit all spaxels and all pixels simultaneously using a GP.
# Then compare the error on both methods to the true model.


# Get the galaxy data in a more useful form, and ignore the spaxels with no flux (e.g., fake spaxels).
has_photons = np.any(np.isfinite(galaxy_flux), axis=-1).flatten()

gp_xy = X[has_photons]
gp_rtheta = X_polar[has_photons]
gp_flux = galaxy_flux.reshape((-1, dispersion.size))[has_photons, :]
gp_flux_err = galaxy_err.reshape((-1, dispersion.size))[has_photons, :]
gp_dispersion = galaxy_dispersion.reshape((-1, dispersion.size))[has_photons, :]

# Use a small number of values.
subset = None
if subset is not None:
    gp_xy = gp_xy[:subset]
    gp_rtheta = gp_rtheta[:subset]
    gp_flux = gp_flux[:subset]
    gp_flux_err = gp_flux_err[:subset]
    gp_dispersion = gp_dispersion[:subset]



import pymc3 as pm

def model_emission_line(gp_dispersion, wavelength, amplitudes, sigma, mean_flux, profile):
    
    # TODO: Don't use true flux value.
    S, P = shape = gp_dispersion.shape
    #rel_emission_flux = amplitudes.reshape((-1, 1)) * profile             
    rel_emission_flux = amplitudes * profile
    rel_emission_flux += 1
    #raise a

    return mean_flux * rel_emission_flux 

# Method 2.
with pm.Model() as model:

    input_dim = 2 # number of latent dimensions for GP (radius, theta)

    mean_flux = tt._shared(flux_mu * np.ones(gp_dispersion.shape)).T
    residual = tt._shared((true_wavelength - gp_dispersion)/true_sigma)
    profile = pm.math.exp(-0.5 * residual**2).T


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
        -\Gamma = -\frac{1}{2l^2}
        \frac{1}{l^2} = 2\Gamma
        l^2 = \frac{1}{2\Gamma}
        l = \sqrt{\frac{1}{2\Gamma}}
    """

    # TODO: Find the periodic lengthscale, don't fit it.
    true_tk_ls = np.sqrt(1/(2 * true_gamma))
    #true_tk_ls = true_gamma/2
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
    # TODO: In future this will get expensive so we will need to produce
    #       a smaller latent space grid. But for now let's just use the
    #       observed spaxal positions in (r, theta).

    # Place a GP prior over the true emission amplitude.
    amplitudes = gp.prior(
        f"amplitude_{true_wavelength:.0f}", 
        X=gp_rtheta,
        reparameterize=False
    )

    # Now we need a function that takes the amplitudes and generates the data.
    y = pm.Normal(
        "y",
        mu=model_emission_line(gp_dispersion, true_wavelength, amplitudes, true_sigma, mean_flux, profile),
        sd=gp_flux_err.T,
        observed=gp_flux.T
    )

    # Optimize.
    p_opt, result = pm.find_MAP(return_raw=True)


def contourf_from_1d_vectors(x, y, z, ax=None, Nx=None, Ny=None, colorbar=False, colorbar_label=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    xgrid = np.linspace(x.min(), x.max(), Nx or np.unique(x).size)
    ygrid = np.linspace(y.min(), y.max(), Ny or np.unique(y).size)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    zgrid = interpolate.griddata(
        (x, y), 
        z, 
        (xgrid, ygrid)
    )

    contours = ax.contourf(xgrid, ygrid, zgrid, antialiased=False)
    if colorbar:
        cbar = plt.colorbar(contours)
        cbar.set_label(colorbar_label)

    return fig


fig = contourf_from_1d_vectors(
    *gp_xy.T, 
    p_opt[f"amplitude_{true_wavelength:.0f}"]
)

mean_flux = (flux_mu * np.ones(gp_dispersion.shape)).T
residual = ((true_wavelength - gp_dispersion)/true_sigma)
profile = np.exp(-0.5 * residual**2).T

# Compare predictions at some places.
opt_flux = model_emission_line(
    gp_dispersion, 
    true_wavelength, 
    p_opt[f"amplitude_{true_wavelength:.0f}"],
    true_sigma, 
    mean_flux,
    profile
).T

idx = 100

fig, ax = plt.subplots()
ax.plot(
    dispersion,
    gp_flux[idx],
    c="k"
)
ax.plot(
    dispersion,
    opt_flux[idx],
    c="tab:red"
)
ax.fill_between(
    dispersion,
    gp_flux[idx] - gp_flux_err[idx],
    gp_flux[idx] + gp_flux_err[idx],
    facecolor="#cccccc",
    zorder=-1
)
ax.set_xlim(true_wavelength - 5, true_wavelength + 5)

# Compare to true values.
true_amplitudes = z[has_photons]

fig = contourf_from_1d_vectors(
    *gp_xy.T,
    p_opt[f"amplitude_{true_wavelength:.0f}"] - true_amplitudes,
    colorbar=True,
    colorbar_label="amplitude residual"
)

'''
# Let's look at the worst fit.
idx = np.argmax(np.abs(p_opt[f"amplitude_{true_wavelength:.0f}"] - true_amplitudes))

fig, ax = plt.subplots()
ax.plot(
    dispersion,
    gp_flux[idx],
    c="k"
)
ax.plot(
    dispersion,
    opt_flux[idx],
    c="tab:red"
)
ax.fill_between(
    dispersion,
    gp_flux[idx] - gp_flux_err[idx],
    gp_flux[idx] + gp_flux_err[idx],
    facecolor="#cccccc",
    zorder=-1
)
ax.set_xlim(true_wavelength - 5, true_wavelength + 5)
'''


'''
# Sampling.
np.random.seed(42)
from time import time
from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull

# Thank god for Dan Foreman-Mackey
def get_step_for_trace(
        trace=None, 
        model=None,
        regular_window=5, 
        regular_variance=1e-3,
        **kwargs
    ):
    model = pm.modelcontext(model)
    
    # If not given, use the trivial metric
    if trace is None:
        potential = QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **kwargs)
        
    # Loop over samples and convert to the relevant parameter space;
    # I'm sure that there's an easier way to do this, but I don't know
    # how to make something work in general...
    samples = np.empty((len(trace) * trace.nchains, model.ndim))
    i = 0
    for chain in trace._straces.values():
        for p in chain:
            samples[i] = model.bijection.map(p)
            i += 1
    
    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)
    
    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    # In the test case for this blog post, this isn't necessary and it
    # actually makes the performance worse so I'll disable it, but I
    # wanted to include the implementation here for completeness
    N = len(samples)
    cov = cov * N / (N + regular_window)
    cov[np.diag_indices_from(cov)] += \
        regular_variance * regular_window / (N + regular_window)
    
    # Use the sample covariance as the inverse metric
    potential = QuadPotentialFull(cov)
    return pm.NUTS(potential=potential, **kwargs)


n_start = 25
n_burn = 500
n_tune = 5000
n_window = n_start * 2 ** np.arange(np.floor(np.log2((n_tune - n_burn) / n_start)))
n_window = np.append(n_window, n_tune - n_burn - np.sum(n_window))
n_window = n_window.astype(int)

regular_window = 1
regular_variance = 1e-2

from tqdm import tqdm

strt = time()
with model:
    start = None
    burnin_trace = None
    for steps in tqdm(n_window):
        step = get_step_for_trace(
            burnin_trace,
            regular_window=regular_window,
            regular_variance=regular_variance
        )
        burnin_trace = pm.sample(
            start=start, 
            tune=steps, 
            draws=2, 
            step=step,
            compute_convergence_checks=False,
            discard_tuned_samples=False
        )
        start = [t[-1] for t in burnin_trace._straces.values()]

    step = get_step_for_trace(
        burnin_trace,
        regular_window=regular_window,
        regular_variance=regular_variance
    )
    dense_trace = pm.sample(
        draws=5000, 
        tune=n_burn, 
        step=step, 
        start=start
    )
    factor = 5000 / (5000 + np.sum(n_window+2) + n_burn)
    dense_time = factor * (time() - strt)

# advi+adapt_diag in left
# advi in right
'''
