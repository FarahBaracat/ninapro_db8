import os
import scipy.io
from metadata import SUBJECT_FILE, DATA_DIR
from src.utils import *
import librosa

pd.set_option('display.max_rows', 1000)

logging.basicConfig(level=logging.INFO)

# ADM parameters
V_THR = 0.5e-5
T_REF = 0.01
ADM_DT = 1 / SAMP_FREQ

GLOVE_CH = 5


def main():
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, SUBJECT_FILE))
    rec_time, n_ch, n_rep, n_stim = extract_rec_info(mat)
    logging.info(f"rec time:{rec_time}   n_ch:{n_ch}   n_stim:{n_stim}  n_rep:{n_rep}")
    emg_df = construct_emg_df(mat, rec_time)
    glove_df = construct_glove_df(mat)

    # extract channels only
    X = emg_df.iloc[:, :n_ch].to_numpy()
    y = glove_df.to_numpy()
    print(f"X (EMG dataset) shape:{X.shape}   target:{y.shape}")

    # Apply RMS on data
    time_window = 0.128  # in second
    overlap = 0
    trial = 1  # trial id starts from 1

    emg_rms, glove_rms, time_bin_ax = get_rms_for_trial(emg_df, glove_df, time_window, overlap, SAMP_FREQ, trial, n_ch,
                                                        index_stim=3)
    emg_rms_entire, glove_rms_entire, _ = get_rms_for_signal(emg_df, glove_df, time_window, overlap, SAMP_FREQ, n_ch,
                                                             index_stim=3)
    print(
        f"emg_rms:{emg_rms.shape}   emg_rms_entire:{emg_rms_entire.shape}")  # ignore/comment if you are using a   #
    # different encoding than ADM  # stim_spikes_model = encode_into_spikes(emg_df,  # n_ch, n_stim, V_THR, T_REF,
    # ADM_DT)  # # Quick Sanity Check  # stim_test = 2  # rep_test = 1  # logging.debug(  #     f"Getting trial {  #
    # rep_test} stim {stim_test} spikes:\n{stim_spikes_model[stim_test][rep_test].trial_times}")  #


def get_rms_for_trial(emg_df, glove_df, time_window, overlap, samp_freq, trial, n_ch, index_stim=3):
    time_win_samples = int(time_window * samp_freq)
    mask = (emg_df['stimulus'] == index_stim) & (emg_df['rerepetition'] == trial)

    index_glove = glove_df[mask].iloc[:, GLOVE_CH]
    index_emg = emg_df[mask].iloc[:, :n_ch]

    # compute number of rms windows
    hop_len, n_win = compute_rms_win_hop(emg_df, index_emg, overlap, time_win_samples)

    time_bin_ax = np.linspace(0, (n_win * time_win_samples) / samp_freq, num=n_win)
    emg_rms = librosa.feature.rms(y=index_emg.T, frame_length=time_win_samples, center=False, hop_length=hop_len)

    glove_rms = librosa.feature.rms(y=index_glove.T, frame_length=time_win_samples, center=False,
                                    hop_length=hop_len).reshape(-1, 1)

    return emg_rms, glove_rms, time_bin_ax


def compute_rms_win_hop(emg_df, index_emg, overlap, time_win_samples):
    if overlap > 0:
        n_win = int((emg_df.shape[0] - time_win_samples) / (overlap * index_emg.shape[0]))
        hop_len = int(time_win_samples * overlap)
    else:
        n_win = int((emg_df.shape[0] - time_win_samples) / (index_emg.shape[0]))
        hop_len = int(time_win_samples)
    print(f"n_win:{n_win}, n_samples:{time_win_samples}")
    return hop_len, n_win


def get_rms_for_signal(emg_df, glove_df, time_window, overlap, samp_freq, n_ch, index_stim=3):
    time_win_samples = int(time_window * samp_freq)
    mask = (emg_df['stimulus'] == index_stim)
    index_glove = glove_df[mask].iloc[:, GLOVE_CH]
    index_emg = emg_df[mask].iloc[:, :n_ch]

    # compute number of rms windows
    hop_len, n_win = compute_rms_win_hop(emg_df, index_emg, overlap, time_win_samples)

    time_bin_ax = np.linspace(0, (n_win * time_win_samples) / samp_freq, num=n_win)
    emg_rms = librosa.feature.rms(y=index_emg.T, frame_length=time_win_samples, center=False, hop_length=hop_len)

    glove_rms = librosa.feature.rms(y=index_glove.T, frame_length=time_win_samples, center=False,
                                    hop_length=hop_len).reshape(-1, 1)

    return emg_rms, glove_rms, time_bin_ax


if __name__ == "__main__":
    main()
