import os
import scipy.io
from metadata import SUBJECT_FILE, DATA_DIR, DATA_PRE_DIR
from src.utils import *
import matplotlib.gridspec as gridspec

pd.set_option('display.max_rows', 1000)

logging.basicConfig(level=logging.INFO)

# ADM parameters
V_THR = 0.5e-5
T_REF = 0.01
ADM_DT = 1 / SAMP_FREQ


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
    time_window = 0.100  # in second
    overlap = 0.025
    trial = 1  # trial id starts from 1

    emg_rms, glove_rms, time_bin_ax = get_rms_for_trial(emg_df, glove_df, time_window, overlap, SAMP_FREQ, trial, n_ch,
                                                        index_stim=3)
    # emg_rms_entire, glove_rms_entire, _ = get_rms_for_signal(emg_df, glove_df, time_window, overlap, SAMP_FREQ, n_ch,
    #                                                          index_stim=3)
    # if  # you  # are  # using a   #  # different encoding than ADM  # stim_spikes_model = encode_into_spikes(  #  #
    # emg_df,  # n_ch, n_stim, V_THR, T_REF,  # ADM_DT)  # # Quick Sanity Check  # stim_test = 2  # rep_test = 1  #
    #  # logging.debug(  #     f"Getting trial {  #  # rep_test} stim {stim_test} spikes:\n{stim_spikes_model[  #  #
    #  stim_test][rep_test].trial_times}")  #
    # emg_rms_entire = emg_rms_entire.reshape(-1, n_ch)

    # save processed data
    emg_rms_df = pd.DataFrame()
    rep_df = pd.DataFrame()
    for trial in range(1, 10, 1):
        emg_rms, glove_rms, time_bin_ax = get_rms_for_trial(emg_df, glove_df, time_window, overlap, SAMP_FREQ, trial,
                                                            n_ch, index_stim=3)
        # print(f"Overlap:{overlap}   emg_rms:{emg_rms.shape}   emg_rms_entire:{emg_rms_entire.shape}")  # #
        # ignore/comment
        emg_rms = emg_rms.reshape(-1, n_ch)
        rep_col = np.repeat(trial, emg_rms.shape[0])
        emg_rms_df = pd.concat([emg_rms_df, pd.DataFrame(emg_rms)])
        rep_df = pd.concat([rep_df, pd.DataFrame(rep_col)])

    emg_rms_df['repetition'] = rep_df[0]

    if not os.path.exists(DATA_PRE_DIR):
        os.makedirs(DATA_PRE_DIR)
    emg_rms_df.to_hdf(os.path.join(DATA_PRE_DIR, f'emg_rms_win_{time_window}_overlap_{overlap}.h5'), key='df', mode='w')

    plot_rms_for_trials(emg_df, glove_df, n_ch, overlap, time_window)  # plt.show()


def plot_rms_for_trials(emg_df, glove_df, n_ch, overlap, time_window):
    for trial in range(1, 10, 1):
        emg_rms, glove_rms, time_bin_ax = get_rms_for_trial(emg_df, glove_df, time_window, overlap, SAMP_FREQ, trial,
                                                            n_ch, index_stim=3)
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(nrows=5, ncols=4)

        for ch in range(n_ch):
            ax = fig.add_subplot(gs[ch])
            ax.plot(time_bin_ax, emg_rms[ch, 0, :])
            ax.title.set_text(f'Ch {ch}')  # plt.title()
        ax = fig.add_subplot(gs[ch + 1])
        ax.plot(time_bin_ax, glove_rms)
        ax.title.set_text(f'Glove Data ')
        fig.suptitle(f"RMS wind:{time_window}   overlap:{overlap}   Trial:{trial}")
        plt.tight_layout()
        plt.show(block=False)


if __name__ == "__main__":
    main()
