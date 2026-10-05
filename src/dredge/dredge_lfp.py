import numpy as np

from .dredgelib import online_displacement, threshold_correlation_matrix
from .motion_util import get_motion_estimate, get_windows


def register_online_lfp(
    lfp_recording,
    rigid=True,
    chunk_len_s=10.0,
    max_disp_um=500,
    # nonrigid window construction arguments
    win_shape="gaussian",
    win_step_um=800,
    win_scale_um=850,
    win_margin_um=None,
    max_dt_s=None,
    # weighting arguments
    mincorr=0.8,
    soft=False,
    # low-level arguments
    thomas_kw=None,
    xcorr_kw=None,
    # misc
    save_full=False,
    device=None,
    pbar=True,
):
    """Online registration of a preprocessed LFP recording

    Arguments
    ---------
    lfp_recording : spikeinterface BaseRecording object
        Preprocessed LFP recording. The temporal resolution of this recording will
        be the target resolution of the registration, so definitely use SpikeInterface
        to resample your recording to, say, 250Hz (or a value you like) rather than
        estimating motion at the original frequency (which may be high).
    rigid : boolean, optional
        If True, window-related arguments are ignored and we do rigid registration
    chunk_len_s : float
        Length of chunks (in seconds) that the recording is broken into for online
        registration. The computational speed of the method is a function of the
        number of samples this corresponds to, and things can get slow if it is
        set high enough that the number of samples per chunk is bigger than ~10,000.
        But, it can't be set too low or the algorithm doesn't have enough data
        to work with. The default is set assuming sampling rate of 250Hz, leading
        to 2500 samples per chunk.
    max_dt_s : float
        Time-bins farther apart than this value in seconds will not be cross-correlated.
        Set this to at least `chunk_len_s`.
    max_disp_um : number, optional
        This is the ceiling on the possible displacement estimates. It should be
        set to a number which is larger than the allowed displacement in a single
        chunk. Setting it as small as possible (while following that rule) can speed
        things up and improve the result by making it impossible to estimate motion
        which is too big.
    win_shape, win_step_um, win_scale_um, win_margin_um : float
        Nonrigid window-related arguments
        The depth domain will be broken up into windows with shape controlled by win_shape,
        spaced by win_step_um at a margin of win_margin_um from the boundary, and with
        width controlled by win_scale_um.
    mincorr : float in [0,1]
        Minimum correlation between pairs of frames such that they will be included
        in the optimization of the displacement estimates.
    device : string or torch.device
        Controls torch device

    Returns
    -------
    me : motion_util.MotionEstimate
        A motion estimate object. me.displacement is the displacement trace, but this object
        includes methods for getting the displacement at different times and depths; see
        the documentation in the motion_util.py file.
    extra : dict
        Dict containing extra info for debugging
    """
    geom = lfp_recording.get_channel_locations()
    fs = lfp_recording.get_sampling_frequency()
    T_total = lfp_recording.get_num_samples()
    T_chunk = min(int(np.floor(fs * chunk_len_s)), T_total)

    # kwarg defaults and handling
    # need lfp-specific defaults
    xcorr_kw = xcorr_kw if xcorr_kw is not None else {}
    thomas_kw = thomas_kw if thomas_kw is not None else {}
    full_xcorr_kw = dict(
        rigid=rigid,
        bin_um=np.median(np.diff(geom[:, 1])),
        max_disp_um=max_disp_um,
        pbar=False,
        device=device,
        **xcorr_kw,
    )
    def weight_fn(Cs, raster, raster_b=None, t_offset_bins=0):
        return threshold_correlation_matrix(
            Cs,
            mincorr=mincorr,
            max_dt_s=max_dt_s,
            in_place=not save_full,
            bin_s=1 / fs,
            t_offset_bins=t_offset_bins,
            soft=soft,
        )

    # get windows
    windows, window_centers = get_windows(
        geom,
        win_step_um,
        win_scale_um,
        spatial_bin_centers=geom[:, 1],
        margin_um=win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )
    P_online, extra = online_displacement(
        lambda t0, t1: lfp_recording.get_traces(start_frame=t0, end_frame=t1).T,
        T_total,
        T_chunk,
        windows,
        geom[:, 1],
        win_scale_um,
        xcorr_kw=full_xcorr_kw,
        weight_fn=weight_fn,
        thomas_kw=thomas_kw,
        save_full=save_full,
        pbar=pbar,
    )
    extra.update(window_centers=window_centers, windows=windows)

    # -- convert to motion estimate and return
    me = get_motion_estimate(
        P_online,
        time_bin_centers_s=lfp_recording.get_times(0),
        spatial_bin_centers_um=window_centers,
    )
    return me, extra
