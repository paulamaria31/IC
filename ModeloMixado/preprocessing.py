import utils

import numpy as np
from scipy.signal import butter, sosfilt, firwin, filtfilt

#Isola uma faixa de frequencia especifica
def bandpass_filter(signal, lowcut, highcut, fs, filter_order, filter_type):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if(filter_type == 'sosfilt'):
        sos = butter(filter_order, [low, high], btype='band', output='sos')
        y = sosfilt(sos, signal)
    elif(filter_type == 'filtfilt'):
        fir_coeff = firwin(filter_order+1,[low,high], pass_zero=False)
        y = filtfilt(fir_coeff, 1.0, signal)

    return y

#Filtra os canais individualmente
def pre_processing(content, lowcut, highcut, frequency, filter_order, filter_type):

    channels = content.shape[0]
    c = 0

    if(filter_type != 'sosfilt' and filter_type != 'filtfilt'):
        print('ERROR: Invalid filter_type parameter. Signal will not be filtered.')
        return content

    while c < channels:
        signal = content[c, :]
        content[c] = bandpass_filter(signal, lowcut, highcut, frequency, filter_order, filter_type)
        c += 1

    return content

#Gerencia a filtragem de todos os sujeitos de uma vez
def filter_data(data, filter, sample_frequency, filter_order, filter_type, verbose = 0):

    filtered_data = list()

    if verbose == 1:
        count = 0
        flag = 0
        print('Data is being filtered: 0%...',end='')

    for signal in data:
        filtered_data.append(pre_processing(signal, filter[0], filter[1], sample_frequency, filter_order, filter_type))

        if verbose == 1:
            count += 1
            flag = utils.verbose_each_10_percent(count, len(data), flag)
    
    return filtered_data

#Normalização do sinal
def normalize_signal(content, normalize_type):

    channels = content.shape[0]
    c = 0
    
    if(normalize_type == 'each_channel'):
        while c < channels:
            content[c] -= np.mean(content[c])
            content[c] += np.absolute(np.amin(content[c]))
            content[c] /= np.std(content[c])
            content[c] /= np.amax(content[c])

            c += 1
    elif(normalize_type == 'all_channels'):
        content -= np.mean(content)

        min_value = np.amin(content)
        while c < channels:
            content[c] += np.absolute(min_value)
            c += 1
        c = 0

        standard_deviation = np.std(content)
        while c < channels:
            content[c] /= standard_deviation
            c += 1
        c = 0

        max_value = np.amax(content)
        while c < channels:
            content[c] /= max_value
            c += 1
        c = 0
    elif(normalize_type == 'sun'):
        while c < channels:
            mean = np.mean(content[c])
            std = np.std(content[c])

            content[c] -= mean
            content[c] /= std

            c += 1
    else:
        print('ERROR: Invalid normalize_type parameter.')

    return content

#Normaliza cada sinal individualmente
def normalize_data(data, normalize_type, verbose = 0):

    normalized_data = list()

    if verbose == 1:
        count = 0
        flag = 0
        print('Data is being normalized: 0%...',end='')

    for signal in data:
        normalized_data.append(normalize_signal(signal, normalize_type))

        if verbose == 1:
            count += 1
            flag = utils.verbose_each_10_percent(count, len(data), flag)

    return normalized_data