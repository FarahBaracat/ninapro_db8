import numpy as np
import logging
import pandas as pd
from metadata import SAMP_FREQ, SUBJECT, COLOR_DICT, STIM_ENCODING, GLOVE_CH
import matplotlib.pyplot as plt
from src.ADM import digitalize_sigma_J
from src.SubjectSpikeModel import *
import librosa


def construct_glove_df(mat):
    glove_df = pd.DataFrame(mat['glove'])

    return glove_df


def construct_emg_df(mat, rec_time):
    emg_df = pd.DataFrame(mat['emg'])
    emg_df['time'] = rec_time
    emg_df['stimulus'] = mat['stimulus']
    emg_df['restimulus'] = mat['restimulus']
    emg_df['rerepetition'] = mat['rerepetition']
    emg_df['trial'] = (emg_df['restimulus'] != emg_df['restimulus'].shift()).cumsum()
    return emg_df


def extract_rec_info(mat):
    n_ch = mat['emg'].shape[1]
    rec_duration = mat['emg'].shape[0] / SAMP_FREQ
    rec_time = np.arange(0, rec_duration, 1 / SAMP_FREQ)
    n_rep = np.unique(mat['rerepetition'])
    n_stim = np.unique(mat['restimulus'])
    logging.info(f"Total recording duration for this subject {SUBJECT}: {rec_duration} seconds")
    logging.info(f"Creating time axis: \nbeg {rec_time[:5]} \nend {rec_time[-5:]}\n")
    logging.info(f"sEMG Channels: {n_ch}")
    return rec_time, n_ch, n_rep, n_stim


def plot_spikes(X, reconst, time, dn_sp, up_sp, ch, rep, adm_vthr, stimulus, adm_t_ref):
    plt.figure()
    plt.plot(time, X, '-', label='original', color=COLOR_DICT['midnight_blue'], linewidth=2)
    plt.plot(time, reconst, '-', label='reconstructed', color=COLOR_DICT['pumpkin'], linewidth=2)
    plt.plot(up_sp, np.repeat(np.max(X), len(up_sp)), '|', markersize=5, color=COLOR_DICT['green_sea'])
    plt.plot(dn_sp, np.repeat(np.max(X), len(dn_sp)), '|', markersize=5, color=COLOR_DICT['pomgrenate'])

    plt.title(f'Ch #{ch}, Trial: {rep}, Gesture:{stimulus} ({STIM_ENCODING[stimulus]})  UP:{adm_vthr}  tref: '
              f'{adm_t_ref}', fontsize=14)
    plt.xlabel('time cropped to 200ms window [s]')
    plt.ylabel(r'Amplitude $\mu V$')
    plt.legend()
    plt.show()

    # plt.savefig(RESULTDIR + 'spike_plots/' + ENG_KEY +'/' + str(RESAMPLE_FREQUENCY) + 'Hz/' + title +  # '.png')


def get_rms_for_trial(emg_df, glove_df, time_window, overlap, samp_freq, trial, n_ch, index_stim=3):
    time_win_samples = int(time_window * samp_freq)
    mask = (emg_df['stimulus'] == index_stim) & (emg_df['rerepetition'] == trial)

    index_glove = glove_df[mask].iloc[:, GLOVE_CH]
    index_emg = emg_df[mask].iloc[:, :n_ch]
    print(f"Single trial: {index_emg.shape}")
    # compute number of rms windows
    hop_len, n_win, time_bin_ax = compute_rms_win_hop(index_emg, overlap, time_win_samples)
    # time_bin_ax = np.arange(0, )
    # time_bin_ax = np.linspace(0, (n_win * time_win_samples) / samp_freq, num=n_win)
    emg_rms = librosa.feature.rms(y=index_emg.T, frame_length=time_win_samples, center=False, hop_length=hop_len)
    glove_rms = librosa.feature.rms(y=index_glove.T, frame_length=time_win_samples, center=False,
                                    hop_length=hop_len).reshape(-1, 1)

    return emg_rms, glove_rms, time_bin_ax


def compute_rms_win_hop(index_emg, overlap, time_win_samples):
    stride = time_win_samples * (1 - overlap)
    # sim_duration_insec = SIM_DURATION / (1000 * ms)
    # stride = sim_duration_insec * (1 - overlap)
    # n_sub_trials = int((TRIAL_DURATION - sim_duration_insec) / stride + 1)
    # sub_trial_start = np.arange(0, (TRIAL_DURATION - sim_duration_insec) + stride, stride)
    # sub_trial_end = sub_trial_start + sim_duration_insec
    #
    # sub_trial_start, sub_trial_end = trim_incomplete_trials(sub_trial_start, sub_trial_end)
    # logging.debug("starting:{}   len:{}".format(sub_trial_start, len(sub_trial_start)))
    # logging.debug("end:{}   len:{}".format(sub_trial_end, len(sub_trial_end)))
    # logging.debug("n_subtrials:{} len(sub_trial_start):{}".format(n_sub_trials, len(sub_trial_start)))
    # assert len(sub_trial_start) == n_sub_trials
    #

    if overlap > 0:
        n_win = int((index_emg.shape[0] - time_win_samples) / stride + 1)  # (overlap * index_emg.shape[0])
        hop_len = int(time_win_samples * (1 - overlap))
        time_bin_ax = np.arange(0, n_win)

    else:
        n_win = int((index_emg.shape[0] / time_win_samples))
        hop_len = int(time_win_samples)  # time_bin_ax = np.arange(0, n_win)
        time_bin_ax = np.arange(0, n_win)

    print(f"n_win:{n_win}, n_samples:{time_win_samples} n_hop:{hop_len}")
    return hop_len, n_win, time_bin_ax


def get_rms_for_signal(emg_df, glove_df, time_window, overlap, samp_freq, n_ch, index_stim=3):
    time_win_samples = int(time_window * samp_freq)
    mask = (emg_df['stimulus'] == index_stim)
    index_glove = glove_df[mask].iloc[:, GLOVE_CH]
    index_emg = emg_df[mask].iloc[:, :n_ch]

    # compute number of rms windows
    hop_len, n_win, time_bin_ax = compute_rms_win_hop(index_emg, overlap, time_win_samples)

    time_bin_ax = np.linspace(0, (n_win * time_win_samples) / samp_freq, num=n_win)
    emg_rms = librosa.feature.rms(y=index_emg.T, frame_length=time_win_samples, center=False, hop_length=hop_len)

    glove_rms = librosa.feature.rms(y=index_glove.T, frame_length=time_win_samples, center=False,
                                    hop_length=hop_len).reshape(-1, 1)

    return emg_rms, glove_rms, time_bin_ax


def encode_into_spikes(emg_df, n_ch, n_stim, v_thr, t_ref, adm_dt):
    stim_spikes_model = {}
    subset_dur = 0.128  # in seconds; for plotting
    subset_samp = int(SAMP_FREQ * subset_dur)
    time_test = np.arange(0, subset_dur, 1 / SAMP_FREQ)

    for stim in n_stim:
        reps = emg_df[(emg_df['restimulus'] == stim)]['rerepetition'].unique()
        list_rep_model = []
        for rep in reps:
            list_ch_model = []
            trial_df = emg_df[(emg_df['restimulus'] == stim) & (emg_df['rerepetition'] == rep)]

            trial_data = trial_df.iloc[:subset_samp, :n_ch].to_numpy()
            logging.debug(f"stim#{stim}   rep:{rep}  trial_df:{trial_data.shape}")
            for ch in range(n_ch):
                trial_reconst, spike_up, spike_dn, spike_time_up, spike_time_dn = digitalize_sigma_J(trial_data[:, ch],
                                                                                                     v_thr, t_ref,
                                                                                                     adm_dt)
                sp_id_up = np.repeat(ch, len(spike_time_up))
                sp_id_dn = np.repeat(ch + n_ch, len(spike_time_dn))
                list_ch_model.append(ChannelSpikesModel(spike_time_up, sp_id_up, spike_time_dn, sp_id_dn))

                if ch == 0 and rep == 1:
                    plot_spikes(trial_data[:, ch], trial_reconst, time_test, spike_time_dn, spike_time_up, ch, rep,
                                v_thr, stim, t_ref)

            list_rep_model.append(TrialSpikesModel(list_ch_model))
        stim_spikes_model[stim] = list_rep_model
    return stim_spikes_model
