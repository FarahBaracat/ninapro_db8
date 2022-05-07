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

rms_time_win = 0.1  # in second
overlap = 0.80


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

    # save processed data
    create_data_dir()
    emg_rms_df = generate_rms_for_index_trials(emg_df, glove_df, n_ch, rms_time_win, overlap)
    emg_rms_df.to_hdf(os.path.join(DATA_PRE_DIR, f'emg_rms_win_{rms_time_win}_overlap_{overlap}.h5'), key='df',
                      mode='w')  # plot_rms_for_trials(emg_df, glove_df, n_ch, overlap, rms_time_win)  # plt.show()


if __name__ == "__main__":
    main()
