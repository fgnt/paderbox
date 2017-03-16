import math
import pyrirgen


def test_1():
    c = 340  # Sound velocity (m/s)
    fs = 16000  # Sample frequency (samples/s)
    r = [2, 1.5, 2]  # Receiver position [x y z] (m)
    s = [2, 3.5, 2]  # Source position [x y z] (m)
    L = [5, 4, 6]  # Room dimensions [x y z] (m)
    rt = 0.4  # Reverberation time (s)
    n = 2048  # Number of samples

    h = pyrirgen.generate_rir(
        L, s, r,
        sound_velocity=c,
        fs=fs,
        reverb_time=rt,
        n_samples=n)

    print(len(h))

    assert len(h) == 2048, len(h)


def test_2():
    c = 340  # Sound velocity (m/s)
    fs = 16000  # Sample frequency (samples/s)
    r = [2, 1.5, 2]  # Receiver position [x y z] (m)
    s = [2, 3.5, 2]  # Source position [x y z] (m)
    L = [5, 4, 6]  # Room dimensions [x y z] (m)
    rt = 0.4  # Reverberation time (s)
    n = 2048  # Number of samples
    mtype = 'omnidirectional'  # Type of microphone
    order = 2  # Reflection order
    dim = 3  # Room dimension
    orientation = 0  # Microphone orientation (rad)
    hp_filter = True  # Enable high-pass filter

    h = pyrirgen.generate_rir(
        L, s, r,
        sound_velocity=c,
        fs=fs,
        reverb_time=rt,
        n_samples=n,
        mic_type=mtype,
        n_order=order,
        n_dim=dim,
        is_high_pass_filter=hp_filter)

    print(len(h))

    assert len(h) == 2048, len(h)


def test_3():
    c = 340  # Sound velocity (m/s)
    fs = 16000  # Sample frequency (samples/s)
    r = [[2, 1.5, 2], [1, 1.5, 2]]  # Receiver position [x y z] (m)
    s = [2, 3.5, 2]  # Source position [x y z] (m)
    L = [5, 4, 6]  # Room dimensions [x y z] (m)
    rt = 0.4  # Reverberation time (s)
    n = 4096  # Number of samples
    mtype = 'omnidirectional'  # Type of microphone
    order = -1  # Reflection order
    dim = 3  # Room dimension
    orientation = 0  # Microphone orientation (rad)
    hp_filter = True  # Enable high-pass filter

    h = pyrirgen.generate_rir(
        L, s, r,
        sound_velocity=c,
        fs=fs,
        reverb_time=rt,
        n_samples=n,
        mic_type=mtype,
        n_order=order,
        n_dim=dim,
        is_high_pass_filter=hp_filter)

    print(len(h), len(h[0]), len(h[1]))

    assert len(h) == 2, len(h)
    assert len(h[0]) == 4096, len(h[0])
    assert len(h[1]) == 4096, len(h[1])


def test_4():
    c = 340  # Sound velocity (m/s)
    fs = 16000  # Sample frequency (samples/s)
    r = [[2, 1.5, 2]]  # Receiver position [x y z] (m)
    s = [2, 3.5, 2]  # Source position [x y z] (m)
    L = [5, 4, 6]  # Room dimensions [x y z] (m)
    rt = 0.4  # Reverberation time (s)
    n = 4096  # Number of samples
    mtype = 'hypercardioid'  # Type of microphone
    order = -1  # Reflection order
    dim = 3  # Room dimension
    orientation = [math.pi / 2, 0]  # Microphone orientation (rad)
    hp_filter = False  # Enable high-pass filter

    h = pyrirgen.generate_rir(
        L, s, r,
        sound_velocity=c,
        fs=fs,
        reverb_time=rt,
        n_samples=n,
        mic_type=mtype,
        n_order=order,
        n_dim=dim,
        is_high_pass_filter=hp_filter)

    print(len(h), len(h[0]))

    assert len(h) == 1, len(h)
    assert len(h[0]) == 4096, len(h[0])

if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
