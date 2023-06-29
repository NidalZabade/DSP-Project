import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


Fs = 8000


def load_echo_path(file_path):
    return sp.io.loadmat(file_path)


def load_speech(file_path):
    return sp.io.loadmat(file_path)


def first_part():
    path = load_echo_path("path.mat")
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


def second_part():
    speech = load_speech("css.mat")

    # the speech signal
    x = speech["css"]
    # the frequency response of the channel
    a = np.ones(1)
    w, h = sp.signal.freqz(x[0, :], a)
    f = w * Fs / (2 * np.pi)

    # plot the speech signal and frequency response of the channel
    # speech signal
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x[0, :])
    plt.title("Speech Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # frequency response
    # plt.subplot(122)
    plt.plot(f, 20 * np.log10(abs(h)))
    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()


def third_part():
    path = load_echo_path("path.mat")
    speech = load_speech("css.mat")

    # the impulse response of the channel
    ht = path["path"]

    # the speech signal
    x = speech["css"]

    # concatenate the speech signal 5 times
    x = np.concatenate((x, x, x, x, x), axis=1)

    # convolve the speech signal with the impulse response of the channel to get the echo signal
    y = np.zeros((1, x.shape[1] + ht.shape[1] - 1))
    for i in range(x.shape[1]):
        y[0, i : i + ht.shape[1]] += x[0, i] * ht[0, :]
    y = y / np.max(np.abs(y))

    # the frequency response of the channel
    a = np.ones(1)
    w, h = sp.signal.freqz(y[0, :], a)
    f = w * Fs / (2 * np.pi)

    # plot the speech signal, echo signal and frequency response of the channel
    plt.figure(figsize=(10, 5))
    # speech signal
    plt.subplot(131)
    plt.plot(x[0, :])
    plt.ylim(-5, 5)
    plt.title("Speech Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # echo signal
    plt.subplot(132)
    plt.plot(y[0, :])
    plt.ylim(-5, 5)
    plt.title("Echo Signal")
    plt.xlabel("Sample Index (n)")
    plt.ylabel("Amplitude")

    # frequency response
    plt.subplot(133)
    plt.plot(f, 20 * np.log10(abs(h)))
    plt.title("Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    plt.tight_layout()
    plt.show()


def main():
    first_part()
    second_part()
    third_part()


if __name__ == "__main__":
    main()
