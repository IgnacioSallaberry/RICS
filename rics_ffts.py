import cv2
import numpy as np

def fftCyclicAutocovariance(signal):
    centered_signal = signal - np.mean(signal)
    ft_signal = np.fft.fftn(centered_signal)
    powerSpectralDensity = np.abs(ft_signal) ** 2
    autocovariance = np.fft.ifftn(powerSpectralDensity) / len(centered_signal)
    return np.real(autocovariance)


def fftCyclicAutocorrelation(signal):
    autocovariance = fftCyclicAutocovariance(signal)
    variance = autocovariance.flat[0]
    if variance == 0.:
        return np.zeros(autocovariance.shape)
    else:
        return (autocovariance / variance)


def fftAutocovariance(signal):
    centered_signal = signal - np.mean(signal)
    padded_shape = [2 * s + 1 for s in centered_signal.shape]
    ft_signal = np.fft.fftn(centered_signal, padded_shape)
    pseudo_powerSpectralDensity = np.abs(ft_signal) ** 2
    pseudo_autocovariance = np.fft.ifftn(pseudo_powerSpectralDensity)
    input_domain = np.ones_like(centered_signal)
    ft_mask = np.fft.fftn(input_domain, padded_shape)
    mask_correction_factors = np.fft.ifftn(np.abs(ft_mask) ** 2)
    autocovariance = pseudo_autocovariance / mask_correction_factors
    crop_slices = [slice(i) for i in signal.shape]
    return np.real(autocovariance[crop_slices])


def fftAutocorrelation(signal):
    autocovariance = fftAutocovariance(signal)
    variance = autocovariance.flat[0]
    if variance == 0.:
        return np.zeros(autocovariance.shape)
    else:
        return (autocovariance / variance)


def cabeza_acf(img):
    power = np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(np.fft.fft2(img))))
    shift = np.fft.fftshift(power)
    normalize = shift / (np.mean(img) * np.mean(img) * len(img[0, :]) * len(img[:, 0])) - 1
    return normalize


def acor_cycle():
    pathlist = Path('/home/ferbellora/Documents/FCS/RICS_rhodamina_9').glob('**/*.tif')
    G = []
    for path in pathlist:
        G.append(cabeza_acf(cv2.imread(path, 0)))
    avg = np.mean(G)
    return avg


def acor_cycle2():
    G = []
    for filename in glob.iglob('/home/ferbellora/Documents/FCS/RICS_rhodamina_9/*.tif'):
        img2 = cv2.imread(filename, 0)
        G.append(cabeza_acf(img2))
    avg = np.mean(G)
    return avg