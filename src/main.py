import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import padasip as pa

Fs = 8000


def load_echo_path(file_path):
    return sp.io.loadmat(file_path)


def load_speech(file_path):
    return sp.io.loadmat(file_path)


def part_one():
    path = load_echo_path("../data/path.mat")
    # the impulse response of the channel
    x = path["path"]

    # the frequency response of the channel
    a = np.ones(1)
    w, h = sp.signal.freqz(x[0, :], a)
    f = w * Fs / (2 * np.pi)

    # plot the impulse response and frequency response of the channel
    # impulse response
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x[0, :])
    plt.title("Impulse Response")
    plt.xlabel("Tap Index (n)")
    plt.ylabel("Amplitude")

    # frequency response
    plt.subplot(122)
    plt.plot(f, 20 * np.log10(abs(h)))
    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()


def part_two():
    speech = load_speech("../data/css.mat")

    # the speech signal
    x = speech["css"]

    # plot the speech signal and frequency response of the channel
    # speech signal
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x[0, :])
    plt.title("Speech Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # Plot the Power Spectral Density (PSD) of the speech signal
    plt.subplot(122)
    plt.psd(x[0, :], NFFT=1024, Fs=Fs)
    plt.title("Power Spectral Density (PSD) of the Speech Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()


def part_three():
    path = load_echo_path("../data/path.mat")
    speech = load_speech("../data/css.mat")

    # the impulse response of the channel
    ht = path["path"]

    # the speech signal
    x = speech["css"]

    # concatenate the speech signal 5 times to get the Far-End signal
    x = np.concatenate((x, x, x, x, x), axis=1)

    # convolve the speech signal with the impulse response of the channel to get the echo signal
    y = np.zeros((1, x.shape[1] + ht.shape[1] - 1))
    for i in range(x.shape[1]):
        y[0, i : i + ht.shape[1]] += x[0, i] * ht[0, :]
    y = y / np.max(np.abs(y))

    # plot the Far-End signal, echo signal and frequency response of the channel
    plt.figure(figsize=(10, 5))
    # Far-End signal
    plt.subplot(131)
    plt.plot(x[0, :])
    plt.ylim(-5, 5)
    plt.title("Far-End Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # echo signal
    plt.subplot(132)
    plt.plot(y[0, :])
    plt.ylim(-5, 5)
    plt.title("Echo Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # Power Density Spectrum (PSD) of echo signal
    plt.subplot(133)
    plt.psd(y[0, :], NFFT=1024, Fs=Fs)
    plt.title("Power Density Spectrum (PSD) of Echo Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()

    # Estimate the power of the Far-End signal and echo signal
    # the formula 10*log10(1/N*sum(x^2)) where N is the length of the signal
    # the power of the Far-End signal
    power_x = 10 * np.log10(1 / (5600 * 5) * np.sum(x**2))
    print(f"the power of the Far-End signal: {power_x}")
    # the power of the echo signal
    power_y = 10 * np.log10(1 / (5600 * 5) * np.sum(y**2))
    print(f"the power of the echo signal: {power_y}")

    # ERL
    ELR = power_x - power_y
    print(f"the ERL: {ELR}")


def part_four():
    path = np.loadtxt("../data/path.txt")
    speech = np.loadtxt("../data/css.txt")

    # Concatenate the speech signal 10 times to get the Far-End signal
    speech = np.concatenate(
        (
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
        ),
        axis=0,
    )

    # Convolve the speech with the path to get the echo signal
    d = np.zeros(len(speech) + len(path) - 1)
    for i in range(len(speech)):
        d[i : i + len(path)] += speech[i] * path
    d = d / np.max(np.abs(d))

    # Inputs to the adaptive filter
    filter_length = 128
    mu = 0.25
    e = np.zeros(len(speech))  # error signal
    y = np.zeros(len(speech))  # estimated echo signal

    # adaptive filter
    adaptive_filter = pa.filters.FilterNLMS(n=filter_length, mu=mu, w="random")

    for i in range(filter_length, len(speech)):
        adaptive_filter.adapt(d[i], speech[i - filter_length : i])
        y[i] = np.dot(adaptive_filter.w, speech[i - filter_length : i])
        e[i] = d[i] - y[i]

    # plot the Far-End signal, echo signal, error signal and the estimated echo signal
    plt.figure(figsize=(10, 5))
    # Far-End signal
    plt.subplot(121)
    plt.plot(speech)
    plt.ylim(-5, 5)
    plt.title("Far-End Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # echo signal
    plt.subplot(122)
    plt.plot(d)
    plt.ylim(-5, 5)
    plt.title("Echo Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    plt.figure(figsize=(10, 5))
    # error signal
    plt.subplot(121)
    plt.plot(e)
    plt.title("Error Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # estimated echo signal
    plt.subplot(122)
    plt.plot(path, label="Echo Path")
    plt.plot(adaptive_filter.w, label="Estimated Echo Path")
    plt.title("Estimated Echo Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.legend()
    plt.show()


def part_five():
    path = np.loadtxt("../data/path.txt")
    speech = np.loadtxt("../data/css.txt")

    # Concatenate the speech signal 10 times to get the Far-End signal
    speech = np.concatenate(
        (
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
        ),
        axis=0,
    )

    # Convolve the speech with the path to get the echo signal
    d = np.zeros(len(speech) + len(path) - 1)
    for i in range(len(speech)):
        d[i : i + len(path)] += speech[i] * path
    d = d / np.max(np.abs(d))

    # Inputs to the adaptive filter
    filter_length = 128
    mu = 0.25
    e = np.zeros(len(speech))
    y = np.zeros(len(speech))

    # adaptive filter
    adaptive_filter = pa.filters.FilterNLMS(n=filter_length, mu=mu, w="random")

    for i in range(filter_length, len(speech)):
        adaptive_filter.adapt(d[i], speech[i - filter_length : i])
        y[i] = np.dot(adaptive_filter.w, speech[i - filter_length : i])
        e[i] = d[i] - y[i]

    # plot the amplitude and phase response for the estimated FIR channel
    plt.figure(figsize=(10, 5))
    # amplitude response
    plt.subplot(121)
    w, h = sp.signal.freqz(path, worN=8000)
    f = w * Fs / (2 * np.pi)
    plt.plot(20 * np.log10(np.abs(h)), label="Path")
    w, h = sp.signal.freqz(adaptive_filter.w, worN=8000)
    f = w * Fs / (2 * np.pi)
    plt.plot(20 * np.log10(np.abs(h)), label="Estimated Path")
    plt.title("Amplitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    # phase response
    plt.subplot(122)
    plt.plot(np.angle(np.fft.fft(path, 1024)))
    plt.plot(np.angle(np.fft.fft(adaptive_filter.w, 1024)))
    plt.title("Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase")
    plt.ylim(-np.pi, np.pi)

    plt.tight_layout()
    plt.legend()
    plt.show()


def part_six():
    path = np.loadtxt("../data/path.txt")
    speech = np.loadtxt("../data/css.txt")

    # Concatenate the speech signal 10 times to get the Far-End signal
    speech = np.concatenate(
        (
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
            speech,
        ),
        axis=0,
    )

    # Convolve the speech with the path to get the echo signal
    d = np.zeros(len(speech) + len(path) - 1)
    for i in range(len(speech)):
        d[i : i + len(path)] += speech[i] * path
    d = d / np.max(np.abs(d))

    # Inputs to the adaptive filter NLMS
    filter_length = 128
    mu = 0.25
    eNLMS = np.zeros(len(speech))
    yNLMS = np.zeros(len(speech))

    # Inputs to the adaptive filter NLMF
    filter_length = 128
    m = 0.005
    eNLMF = np.zeros(len(speech))
    yNLMF = np.zeros(len(speech))

    # adaptive filter NLMS
    adaptive_filter1 = pa.filters.FilterNLMS(
        n=filter_length, mu=mu, eps=1e-6, w="random"
    )
    # adaptive filter NLMF
    adaptive_filter2 = pa.filters.FilterNLMF(n=filter_length, mu=m, w="random")

    for i in range(filter_length, len(speech)):
        adaptive_filter1.adapt(d[i], speech[i - filter_length : i])
        adaptive_filter2.adapt(d[i], speech[i - filter_length : i])
        yNLMS[i] = np.dot(adaptive_filter1.w, speech[i - filter_length : i])
        yNLMF[i] = np.dot(adaptive_filter2.w, speech[i - filter_length : i])
        eNLMS[i] = d[i] - yNLMS[i]
        eNLMF[i] = d[i] - yNLMF[i]

    # error signal for NLMS and LMS
    plt.subplot(121)
    plt.plot(eNLMS, label="Error Signal NLMS")
    plt.plot(eNLMF, label="Error Signal NLMF")
    plt.title("Error Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # estimated echo signal for NLMS and LMS
    plt.subplot(122)
    plt.plot(path, label="Echo Path")
    plt.plot(adaptive_filter1.w, label="Estimated Echo Path NLMS")
    plt.plot(adaptive_filter2.w, label="Estimated Echo Path NLMF")
    plt.title("Estimated Echo Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.legend()
    plt.show()


def main():
    # part_one()
    # part_two()
    # part_three()
    # part_four()
    # part_five()
    # part_six()
    pass


if __name__ == "__main__":
    main()
