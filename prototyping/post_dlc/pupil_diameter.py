import numpy as np
from one.api import ONE
from scipy.interpolate import interp1d


def get_dlc_XYs(one, eid, view, likelihood_thresh=0.9):
    dataset_types = ['camera.dlc', 'camera.times']
    try:
        times = one.load_dataset(eid, '_ibl_%sCamera.times.npy' % view)
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view)
    except KeyError:
        print('not all dlc data available')
        return None, None
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y]).T
    return times, XYs





def smooth_interpolate_signal_sg(signal, window=31, order=3, interp_kind='cubic'):
    """Run savitzy-golay filter on signal, interpolate through nan points.
    
    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'

    Returns
    -------
    np.array
        smoothed, interpolated signal for each time point, shape (t,)
        
    """

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x.

    This is based on 
    https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    https://dsp.stackexchange.com/a/64313

    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed



def get_pupil_diameter(XYs):
    """Estimate pupil diameter by taking median of different computations.
    
    In the two most obvious ways:
    d1 = top - bottom, d2 = left - right
    
    In addition, assume the pupil is a circle and estimate diameter from other pairs of 
    points
    
    Author: Michael Schartner
    
    Parameters
    ----------
    XYs : dict
        keys should include `pupil_top_r`, `pupil_bottom_r`, 
        `pupil_left_r`, `pupil_right_r`

    Returns
    -------
    np.array
        pupil diameter estimate for each time point, shape (n_frames,)
    
    """
    
    # direct diameters
    t = XYs['pupil_top_r'][:, :2]
    b = XYs['pupil_bottom_r'][:, :2]
    l = XYs['pupil_left_r'][:, :2]
    r = XYs['pupil_right_r'][:, :2]

    def distance(p1, p2):
        return ((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2) ** 0.5

    # get diameter via top-bottom and left-right
    ds = []
    ds.append(distance(t, b))
    ds.append(distance(l, r))

    def dia_via_circle(p1, p2):
        # only valid for non-crossing edges
        u = distance(p1, p2)
        return u * (2 ** 0.5)

    # estimate diameter via circle assumption
    for side in [[t, l], [t, r], [b, l], [b, r]]:
        ds.append(dia_via_circle(side[0], side[1]))
    diam = np.nanmedian(ds, axis=0)


    return diam


def get_raw_and_smooth_pupil_dia(eid, video_type):

    # likelihood threshold
    l_thresh = 0.9

    # camera view
    view = video_type

    # threshold (in standard deviations) beyond which a point is labeled as an outlier
    std_thresh = 5

    # threshold (in seconds) above which we will not interpolate nans, but keep them
    # (for long stretches interpolation may not be appropriate)
    nan_thresh = 1

    # compute framerate of camera
    if view == 'left':
        fr = 60  # set by hardware
        window = 31  # works well empirically
    elif view == 'right':
        fr = 150  # set by hardware
        window = 75  # works well empirically
    else:
        raise NotImplementedError

    # load markers
    _, markers = get_dlc_XYs(one, eid, view, likelihood_thresh=l_thresh)

    # compute diameter using raw values of 4 markers (will be noisy and have missing data)
    diam0 = get_pupil_diameter(markers)

    # run savitzy-golay filter on non-nan timepoints to denoise
    diam_sm0 = smooth_interpolate_signal_sg(
        diam0, window=window, order=3, interp_kind='linear')

    # find outliers, set to nan
    errors = diam0 - diam_sm0
    std = np.nanstd(errors)
    diam1 = np.copy(diam0)
    diam1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    diam_sm1 = smooth_interpolate_signal_sg(
        diam1, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(1 * np.isnan(diam1))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            diam_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff
            
    # diam_sm1 is the final smoothed pupil diameter estimate
    return diam0, diam_sm1


def SNR(diam0, diam_sm1):

    # compute signal to noise ratio between raw and smooth dia
    good_idxs = np.where(~np.isnan(diam_sm1) & ~np.isnan(diam0))[0]
    snr = (np.var(diam_sm1[good_idxs]) / 
           np.var(diam_sm1[good_idxs] - diam0[good_idxs]))
           
    return snr



if __name__ == "__main__":    

    '''one pqt file per camera, e.g. _ibl_leftCamera.features.pqt 
    and it will contain columns named in Pascal case, 
    the same way you would name an ALF attribute, e.g. pupilDiameter_raw and 
    lick_times.'''

    one = ONE()    
    eid = '572a95d1-39ca-42e1-8424-5c9ffcb2df87'
    video_type = 'left'
    
    pupil_dia_raw_left, pupil_dia_smooth_left = (
        get_raw_and_smooth_pupil_dia(eid, video_type))
    
    print('SNR left',SNR(pupil_dia_raw_left, pupil_dia_smooth_left))     
    
    video_type = 'right'
    pupil_dia_raw_right, pupil_dia_smooth_right = (
        get_raw_and_smooth_pupil_dia(eid, video_type))        

    print('SNR right',SNR(pupil_dia_raw_right, pupil_dia_smooth_right))

