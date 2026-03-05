import numpy as np
from getf0Quantile import getQ

def soft_clamp_f0(f0, f0_min, f0_max=None, alpha=0.5):
    f0 = f0.copy()

    # clamp cao nếu cần
    if f0_max is not None:
        f0[f0 > f0_max] = f0_max

    # soft clamp thấp
    low = f0 < f0_min
    f0[low] = f0_min * (f0[low] / f0_min) ** alpha

    return f0

def soft_clamp_log(f0, f0_min, f0_max=None, beta=0.5, semitone=-1):
    f0 = f0.copy()

    # clamp cao
    if f0_max is not None:
        f0[f0 > f0_max] = f0_max

    # mask frame thấp
    low = f0 < f0_min
    if np.any(low):
        ratio = f0[low] / f0_min
        f0_low = f0_min * np.exp(beta * np.log(ratio + 1e-8))  # soft clamp
        #f0_low = pitch_shift(f0_low, semitone)  # shift xuống
        f0[low] = f0_low

    return f0

def pitch_shift(f0, semitone):
    return f0 * (2 ** (semitone / 12))

def compress_low_f0(f0, LOG_HZ_MAX, LOG_HZ_MIN, gammaLow=0.25, gammaHigh =0.5):
    f0 = f0.astype(np.float32, copy=False)
    voiced = f0 > 0
    if not np.any(voiced):
        return f0

    log_f0 = np.log(f0[voiced] + 1e-6)

    # low
    low = log_f0 < LOG_HZ_MIN
    log_f0[low] = LOG_HZ_MIN + gammaLow * (log_f0[low] - LOG_HZ_MIN)

    # high
    high = log_f0 > LOG_HZ_MAX
    log_f0[high] = LOG_HZ_MAX + gammaHigh * (log_f0[high] - LOG_HZ_MAX)

    f0[voiced] = np.exp(log_f0)
    return f0


def hz_extrapolate_limit(f_ref_hz, semitone_down):
    return f_ref_hz * (2 ** (-semitone_down / 12))

def shift_f0_contour(
    f0,
    HZ_MIN,
    HZ_MAX,
    max_semitone=5,
    safety_semitone=2,
    manual_semitone = None
):
    f0 = f0.astype(np.float32, copy=False)
    voiced = f0 > 65
    if not np.any(voiced):
        return f0

    med = np.median(f0[voiced])
    semitone = 0.0
    high = f0 > HZ_MAX
    low = (f0 < HZ_MIN) & voiced

    if med < HZ_MIN:
        print("LOW:")
        mean_low = np.mean(f0[low])
        print(mean_low)
        semitone = 12 * np.log2(HZ_MIN / mean_low)
    elif med > HZ_MAX:
        print("HIGH:")
        mean_high = np.mean(f0[high])
        print(mean_high)
        semitone = 12 * np.log2(HZ_MAX / mean_high)
    # high_mean = np.mean(f0[high]) if np.any(high) else 0
    # low_mean = np.mean(f0[low]) if np.any(low) else 0
    # semitone_low = 12 * np.log2(HZ_MIN / low_mean) if low_mean!=0 else 0
    # semitone_high = 12 * np.log2(HZ_MAX / high_mean) if high_mean!=0 else 0
    # semitone = semitone_low + semitone_high

    # clamp theo khả năng extrapolate của model
    if manual_semitone is not None:
        semitone = manual_semitone
    else:
        semitone = np.clip(semitone, -max_semitone, max_semitone)
    print(semitone)
    # print(semitone_high)
    # print(semitone_low)
    print(med)
    print(HZ_MIN)
    print(HZ_MAX)

    if abs(semitone) < 0.1:
        return f0

    f0[voiced] *= 2 ** (semitone / 12)

    # high = f0 > HZ_MAX
    #
    # if np.any(high):
    #     mean_high = np.mean(f0[high])
    #     print(mean_high)
    #     corr_semitone = 12 * np.log2(HZ_MAX / mean_high)
    #     f0[high] *= 2 ** (corr_semitone / 12)
    #     print(corr_semitone)
    #
    # # lower side
    # hz_min_limit = HZ_MIN * 2 ** (-safety_semitone / 12)
    # low = (f0 < hz_min_limit) & voiced
    #
    # if np.any(low):
    #     mean_low = np.mean(f0[low])
    #     corr_semitone = 12 * np.log2(HZ_MIN / mean_low)
    #     f0[low] *= 2 ** (corr_semitone / 12)
    #     print(corr_semitone)


    return f0

def shift_f0_contourB(
    f0,
    HZ_MIN,
    HZ_MAX,
    max_semitone=10
):
    #f0 = f0.astype(np.float32, copy=False)
    voiced = f0 > 65
    if not np.any(voiced):
        return f0
    med = np.median(f0[voiced])
    semitone = 0.0
    if med < HZ_MIN:
        semitone = 12 * np.log2(HZ_MIN / med)
    elif med > HZ_MAX:
        semitone = 12 * np.log2(HZ_MAX / med)

    # clamp theo khả năng extrapolate của model
    semitone = np.clip(semitone, -max_semitone, max_semitone)
    print(semitone)
    print(med)
    print(HZ_MIN)
    print(HZ_MIN*0.8)
    print(HZ_MAX)

    if abs(semitone) < 0.1:
        return f0

    f0[voiced] *= 2 ** (semitone / 12)
    return f0