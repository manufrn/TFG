import numpy as np
import matplotlib.pyplot as plt
import pycwt
import sys
from datetime import datetime, timedelta
from analysis_functions import *

temp, _, time, _, _ = load_time_series('AGL_1_SB56.h5', convert_date=False)

time = time[:60000]
temp = temp[8, :60000]
N = temp.size
dt = 5
print(N/2)

def detrend_and_normalize(temp, time):
    p = np.polyfit(time - time[0], temp, 1)
    temp_notrend = temp - np.polyval(p, time - time[0])
    std = temp_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    temp_norm = temp_notrend / std

    return temp_norm


def power_of_two(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                return 2 ** i
    else:
        return 1


# detrend and normalize data
p = np.polyfit(time - time[0], temp, 1)
data_notrend = temp - np.polyval(p, time - time[0])
std = data_notrend.std()  # Standard deviation
var = std ** 2  # Variance
data_norm = data_notrend / std
data_norm = np.pad(data_norm, (0, power_of_two(N) - N))

# define parametres
mother = pycwt.Morlet(12)
s0 = 2 * dt
dj = 0.5
# J = 7 / dj
J = int(1/dj * np.log2(N*dt/s0))

alpha, _, _ = pycwt.ar1(temp)

# wavelet transform
wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(data_norm, dt, dj, s0, J, mother)
power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1 / freqs
                                                      
power = power[:, :N]
coi = coi[:N]

# significance 
signif, fft_theor = pycwt.significance(1.0, dt, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95
glbl_power = power.mean(axis=1)
dof = power_of_two(N) - scales  # Correction for padding at edges
glbl_signif, tmp = pycwt.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)

print(power.shape)
print(coi.shape)
# plotting
time = np.array([datetime.fromtimestamp(i) for i in time])
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
ax1.plot(time, temp, 'k')

levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 6, 16]
ax2.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
            extend='both', cmap='viridis')
extent = [time.min(), time.max(), min(period), max(period)]
# ax2.contour(time, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
#            extent=extent)
ax2.set_ylim(max(np.log2(period)), min(np.log2(period)))
t = time
dt = timedelta(seconds=dt)
ax2.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                           np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')

Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                           np.ceil(np.log2(period.max())))
ax2.set_yticks(np.log2(Yticks))
ax2.set_yticklabels(Yticks)
fig.tight_layout()
plt.show()
