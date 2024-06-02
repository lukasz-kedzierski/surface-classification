def select_sequence(series, selected_columns, window_length, freq_stride):
    series = series[selected_columns]
    init_timestep = random.randint(0, (len(series) - 1) - window_length)
    series = series.iloc[init_timestep:init_timestep + window_length:freq_stride]
    return series
    
def get_sample_features(window):
    "Get features for single sequence"
    
    window_df = pd.DataFrame()
    print(type(window))
    for col in window.columns:
        time_features = get_time_domain(window[col], col)
        ft_features = get_frequency_domain(window[col], col)
        features = pd.concat([time_features, ft_features], axis=1)
        window_df = pd.concat([window_df, features], axis=1)
    return window_df

def get_time_domain(window, feature_name):
    " How the signal changes with a time"

    col = feature_name
    columns = [f'{col}_min', f'{col}_max', f'{col}_mean', f'{col}_std', f'{col}_skewness', f'{col}_kurtosis', 
               f'{col}_rms', f'{col}_peak', f'{col}_peak_to_peak', f'{col}_crest_factor', f'{col}_form_factor', f'{col}_pulse_indicator']
    time_window = pd.DataFrame(columns=columns)
    
    min = np.min(window)
    max = np.max(window)
    mean = np.mean(window)
    std = np.std(window)
    skewness = stats.skew(window)
    kurtosis = stats.kurtosis(window)

    # # other features
    rms = np.sqrt(np.mean(window**2))
    peak = np.max(np.abs(window))
    peak_to_peak = np.ptp(window) # the range between minimum and maximum values

    crest_factor = np.max(np.abs(window))/np.sqrt(np.mean(window**2)) # how extreme the peaks are in a waveform
    form_factor = np.sqrt(np.mean(window**2))/np.mean(window) # the ratio of the RMS (root mean square) value to the average value
    pulse_indicator = np.max(np.abs(window))/np.mean(window)

    features = [[min, max, mean, std, skewness, kurtosis, 
                 rms, peak, peak_to_peak, crest_factor, form_factor, pulse_indicator]]
    
    return pd.concat([pd.DataFrame(features, columns=time_window.columns), time_window], ignore_index=True)

def get_frequency_domain(window, feature_name):
    "How much of the signal lies within each given frequency band over a range of frequencies"
    
    col = feature_name
    columns = [f'{col}_ft_sum', f'{col}_ft_max', f'{col}_ft_mean', f'{col}_ft_peak', f'{col}_ft_variance']
   
    time_window = pd.DataFrame(columns=columns)

    ft = fft(window.values)
    S = np.abs(ft**2)/len(window)
    
    ft_sum = np.sum(S)
    ft_max = np.max(S)
    ft_mean = np.mean(S)    
    ft_peak = np.max(np.abs(S))
    ft_variance = np.var(S)

    features = [[ft_sum, ft_max, ft_mean, ft_peak, ft_variance]]
    
    return pd.concat([pd.DataFrame(features, columns=time_window.columns), time_window], ignore_index=True)