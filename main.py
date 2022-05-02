import os
import scipy.io
from metadata import subject_dataset
from src.utils import *
from src.ADM import digitalize_sigma_J
# from src.ADM import ADM

pd.set_option('display.max_rows', 1000)

data_dir = 'data/'
subject_file = f'S{subject}_E1_A{subject_dataset}.mat'  # S1_E1_A1.mat

logging.basicConfig(level=logging.DEBUG)

V_THR = 0.3e-5
T_REF = 0.0001
ADM_DT = 1 / SAMP_FREQ


def main():
    mat = scipy.io.loadmat(os.path.join(data_dir, subject_file))
    rec_time, n_ch = extract_rec_info(mat)

    emg_df = construct_emg_df(mat, rec_time)
    glove_df = construct_glove_df(emg_df, mat, rec_time)

    # extract channels only
    X = emg_df.iloc[:, :n_ch].to_numpy()
    y = glove_df.to_numpy()
    print(f"X (EMG dataset) shape:{X.shape}   target:{y.shape}")  # spikify  # spike_conversion(X, y, V_THR, T_REF, dt)

    # test on subset
    ch_test = 0

    trial_data = X[:200, ch_test]
    time_test = np.arange(0, 0.1, 1 / SAMP_FREQ)
    trial_reconst, spike_up, spike_dn, spike_time_up, spike_time_dn = digitalize_sigma_J(trial_data, V_THR, T_REF,
                                                                                         ADM_DT)

    # Using numba
    # t_up, t_dn, times_interpolated, spike_idx_up, spike_idx_dn = ADM(trial_data, V_THR, V_THR, SAMP_FREQ, T_REF,
    #                                                                  return_indices=True, index_dt=1e-4)

    plot_spikes(trial_data, trial_reconst, time_test, spike_time_dn, spike_time_up)


if __name__ == "__main__":
    main()