import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime, timezone


def parse_arguments():

    print("Display choice:")
    detector_choices = ["NaITl", "BGO", "CdTe"]
    detector = input(f"Detector choice in {detector_choices}: ")

    display_choices = ["Spectrum", "Calibration", "Resolution", "Efficiency", \
                       "Geometric factor", "Off-axis response"]
    function_display = input(f"Choice of display in {display_choices}: ")

    args = {'detector': detector, 'function_display': function_display}

    angles = [0, 30, 60, 90]

    if function_display == "Spectrum":
        sources = ["americium", "barium", "caesium", "cobalt", "all"]
        source = input(f"Source choice in {sources}: ")

        angle = input(f"Angle choice in {angles}° (default 0°): ")
        if angle == "":
            angle = '0'

        args.update({'source': source, 'angle': angle})

    elif function_display == "Off-axis response":
        param = ["Resolution", "Efficiency"]

        param_choice = input(f"Parameter to display between {param}: ")
        A = 'ε = f(E) for different angles'
        B = 'ε = f(θ) for different energies'
        function_choice = input(f"Function choice between A: {A} and B: {B}:")

        args.update({'function_choice': function_choice, 'param_choice': \
                     param_choice})

    elif function_display in ["Resolution", "Efficiency"]:
        angle = input(f"Angle choice in {angles} (default 0°): ")
        if angle == "":
            angle = '0'
        comparison = input(f"Detector comparison on the same graph (y/n)?")

        args.update({'angle': angle, 'comparison': comparison})

    return args

# --------------------------------- FUNCTIONS ---------------------------------

def only_data(data_list):
        """ 
        remove all the information prior to the data in the list obtained from
        the datasheets of the detectors and gives the exposure time
        """
        new_data = []
        i = 0
        while data_list[i] != '0':   
        # go through the lines until we meet the first data
            if data_list[i] == '$MEAS_TIM:':    # indicator of the exposure 
                                # time for NaITl and BGO detectors
                times = data_list[i+1].split()
                exposure_time = int(times[0])   # we consider the 'real time'
            elif data_list[i][:9] == 'REAL_TIME':   # indicator of the exposure 
                                    # time for the CdTe detector
                exposure_time = int(data_list[i][12:15])
            i += 1
        while data_list[i] != '<<END>>' and data_list[i] != '$ROI:':
            # get all the data (need to stop before more information are listed
            # at the end of the datasheet -and thus the list of data).
            new_data.append(float(data_list[i]))
            i += 1
        return [new_data, exposure_time]

def openfile(source, angle, detector):
    """ 
    returns two lists: the count rate and the associated channels
    """
    # first, we need to write the file name as the user only gives the source,
    # detector and angle
    if detector == 'CdTe':  # CdTe files are in a different format
        filename = f'{source}_{angle}_CdTe.mca'
    else:
        filename = f'{source}_{angle}_{detector}.Spe'
    
    data = []
    with open(filename, 'r', errors='ignore') as file:
        for line in file:
            data.append(line.strip())

    # from all the information in the file, we only keep the exposure time and
    # of course the counts
    data, exposure_time = only_data(data)
    count_rate = [element/ exposure_time for element in data] # transform the counts 
                                                        # into count rates
    count_rate_err = [np.sqrt(element) / exposure_time for element in data]
    channels = [i for i in range(len(data))]
    
    return count_rate, channels, count_rate_err

def remove_background(count_rate, detector):
    """ 
    returns the list of count rates without the count rates of the background
    (taken with the same detector, but not the same exposure time, hence the
    need to use the count rate instead of the counts)
    """
    data_background = openfile('background', '0', detector)
    background_err = data_background[2]  
    background_count_rate = data_background[0]  # get the count rate from the 
                                                # background 
    # subtracting from our original dataset the background count rate
    data_without_b = [count_rate[i] - background_count_rate[i] for i in range(\
        len(count_rate))]

    return data_without_b, background_err

def gaussian_model(x, mu, sig, A):
    """
    returns a gaussian function 
    A: amplitude not normalised
    μ: centroïd
    σ: FWHM

    """
    return A * np.exp( - (x - mu) ** 2 / (2 * sig ** 2) ) 

def polynomial(x, a, b, c):
    """ 
    returns a polynomial function
    """
    return a * x ** 2 + b * x + c

def format_result(params, popt, pcov, precision = 4):
    """
    Displays parameter best estimates and uncertainties with a precision equal
    to 4 decimals by delfault
    """
    perr = np.sqrt(np.diag(pcov))
    
    _lines = (f"{p} = {round(o, precision)} ± {round(e, precision)}" for p, o, 
              e in zip(params, popt, perr))
    
    return "\n".join(_lines)

def initial_parameters(list, x_min, x_max):
    """
    returns guessed initial parameters for the gaussian function

    here we set the amplitude A as the maximum value of the peak, the μ value 
    corresponding to the point on which the peak is centered is set as the 
    interval center, and σ is the FWHM of the peak (up to a factor) set to 
    half of the interval length.
    """
    maxx = 0
    for i in list:
        if i > maxx:
            maxx = i
    A = maxx
    mu = (x_min + x_max)/2
    FWHM = (x_max - x_min) / 2 
    sig = FWHM / 2 * np.sqrt(2* np.log(2))

    return [mu, sig, A]

def photopeak_interval(detector, source, CdTe_resolution):
    """ 
    returns all the intervals in which you can find a photopeak.
    I wrote these intervals in the photopeak_interval.txt file for each source 
    whith each after observing the spectra.
    """
    filename = 'photopeak_intervals.txt'
    photopeak = []
    with open(filename, 'r') as file:
        for line in file:
            photopeak.append(line.split())
    # find the information for the given detector and source
    i = 0
    while photopeak[i][0] != detector: 
        i += 1
    while photopeak[i][0] != source:
        i += 1
    # converts the string object from the datasheet to a number
    interest_region = int(photopeak[i][1])
    k = 2
    intervals = []
    while k < len(photopeak[i]):
        if photopeak[i][k] != 'other:': # this condition is because of the CdTe
            # detector which does't have enough photopeaks for the resolution,
            # so other peaks intervals are defined ather the 'others:' 
            # indication. 
            intervals.append(float(photopeak[i][k]))
            k += 1
        else:
            if CdTe_resolution: # if we are using the photopeak_interval
                # function for resolution with CdTe, we are going to need the
                # values put after the indication 'others:'
                k += 1
            else:   # if not, then all photopeaks are already listed, no need 
                    # for other peaks
                k = len(photopeak[i])

    return intervals, interest_region

def sup_values(datalists_set):
    """ 
    returns a list which values are the sum of the values of the same index 
    of all the lists of the datalists_set
    """
    new_data = []
    for i in range(len(datalists_set[0])):
        value = 0
        for j in range(len(datalists_set)):
            value += datalists_set[j][i]
        new_data.append(value)

    return new_data

def gaussian_fit(detector, source, angle, plot_display = True, \
                 CdTe_resolution=False):
    """ 
    returns the parameters of the gaussian models of all photopeaks of the
    specified source observed with the specified detector at a specified angle
    or displays the spectrum for these same specified parameters
    Note that source can be 'all', and the function will return one single 
    figure with all four spectra displayed (on four graphs)
    """
    
    # two situations possible: user specified a source or wrote 'all'. This 
    # choice leads to two different figure format:
    if source == 'all':
        if plot_display:
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            fig_title = f'Spectra for all sources with the {detector} detector'
            fig.suptitle(fig_title, fontweight='bold', fontsize=17)
            plt.subplots_adjust(hspace=0.3, wspace=0.2)
            ax = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
        sources = ['americium', 'barium', 'caesium', 'cobalt']
    else:
        if plot_display:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax = [ax]   # ax needs to be a list to match the type of ax of the 
                        # 'all' situation ; same for the source
        sources = [source]
    
    for source in sources:  # for each of the 4 sources if 'all' situation, or
                            # only the specified source

        data, channels, source_uncertainties = \
            openfile(source, angle, detector)   # have the information on the 
                                                # source 
        background_data = remove_background(data, detector) # have the 
            # information about the background acquisition with the detector

        if detector == 'CdTe':
            uncertainties = [0.000001 for i in range(len(data))]    # for the 
                # CdTe detector, I don't subtract the background, because the
                # peaks are not visible otherwise, so the uncertainty should 
                # be the uncertainty on the source, but as the count rate is
                # already quite poor, when adding uncertainties, the curve_fit
                # function doesn't find any solution. So I added a very little
                # uncertainty of 10^-6 to follow the same scheme as the other
                # detectors later in the code.
        else:
            background_uncertainties = background_data[1]   # extraction of the
                                        # background uncertainties
            uncertainties = [np.sqrt(s_u ** 2 + b_u ** 2) for (s_u, b_u) in \
                             zip(source_uncertainties, \
                                 background_uncertainties)] # use error 
                            # propagation to find the uncertainty on the count
                            # rate with background subtracted.

        if detector != 'CdTe' :
            no_background_data = background_data[0]
            data = np.asarray(no_background_data)   # background subtracted
                    # for all detectors but CdTe
        else:
            data = np.asarray(data)

        channels = np.asarray(channels)
        
        # extraction of all photopeaks interval for the source on this detector
        photopeaks_info = photopeak_interval(detector, source, CdTe_resolution)
        intervals = photopeaks_info[0]
        z = photopeaks_info[1]  # gives the region of interest in the spectrum

        if plot_display:    # plot the data with their uncertainties
            ax[sources.index(source)].errorbar(channels[:z], data[:z], \
                                        yerr=uncertainties[:z], fmt='+', \
                                        label='raw data with uncertainties', \
                                        color='gray', zorder=1, capsize=5)
        
        peaks_data = [] # will contain all information about the photopeaks for
                        # the specified source, angle, detector.

        # these four lists will contain the parameters values for each of the 
        # photopeak of the source, and their error will be in the last list:
        centre_positions, fwhm, amplitudes, errors = [], [], [], []
        nb_peaks = 1
        
        if intervals != []: # if there is any photopeak on the spectrum, we 
                            # need to implement a gaussian model on it
            
            for i in range(0, len(intervals), 2):   # for each photopeak (the 
                # list of intervals gives the minimum and maximum of all 
                # intervals one after the other, hence the step of 2 when 
                # browsing this list)

                # definition of the interval:
                channel_min, channel_max = intervals[i], intervals[i+1]

                # selection of the data (count rate, channels and uncertainties
                # arrays) only in the photopeak interval:
                data_photopeak = data[int(channel_min):int(channel_max)]
                channels_photopeak = channels[int(channel_min): \
                                              int(channel_max)]
                channel_uncertainties = uncertainties[int(channel_min): \
                                                      int(channel_max)]

                # set the initial parameters for the gaussian model
                initial_param = initial_parameters(data_photopeak, \
                                            int(channel_min), int(channel_max))

                # finding the model with curve_fit function, tanking into 
                # account the uncertainties on the count rate, which will 
                # weight the fit with 'absolute_sigma' set:
                popt, pcov = curve_fit(gaussian_model, channels_photopeak, \
                                       data_photopeak, p0=initial_param, \
                                        sigma=channel_uncertainties, \
                                        absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                
                
                peaks_data.append(gaussian_model(channels, *popt))
                
                if plot_display:    # plot the photopeak fitted with the model
                    label_text = f'gaussian fit of peak {nb_peaks}'
                    ax[sources.index(source)].plot(\
                        channels[:z], gaussian_model(channels[:z], *popt), \
                            '--', label=label_text)
                    popt[1] = abs(popt[1])  # insure the FWHM (up to a factor)
                                            # is positive. (it does not change
                                            # the model to do so because the
                                            # intervention of the FWHM in the
                                            # gaussian function is only to the 
                                            # square).
                    # display the parameters of each photopeak gaussian model 
                    # with their uncertainties
                    params = ('µ', 'σ', 'A')
                    print('\n'+ label_text + '\n', \
                          format_result(params, popt, pcov))
                
                # these condition is to avoid the problem of having a peak that
                # is not found by the curve_fit function, such as the photopeak
                # of Cobalt with NaITl at 30°, because the uncertainty on this
                # peak will be so bad that the centroid can be negative...
                if popt[0] > 0:
                    errors.append(perr)
                    centre_positions.append(round(popt[0], 4))
                    fwhm.append(abs(round(popt[1], 4)))
                    amplitudes.append(round(popt[2], 4))

                data = data - gaussian_model(channels, *popt)
                nb_peaks += 1

            
            if plot_display:    # plot the whole spectrum, summing all 
                                # photopeak models
                ax[sources.index(source)].plot(channels[:z], sup_values(peaks_data)[:z], color='crimson', label='spectrum')
    
        if plot_display:    # set the parameters of the figure(s)
            plot_title = f'{source} spectrum with {detector} detector at {angle}°'
            plot_title = plot_title[0].upper() + plot_title[1:]
            if sources != ['source']:   # = if 'all' situation ; we need to set
                                        # a title to all graphs.
                ax[sources.index(source)].set_title(plot_title, size=15)
                ax[sources.index(source)].set_xlabel('channel')
                ax[sources.index(source)].set_ylabel('counts/s')
                ax[sources.index(source)].legend()
            else:
                plt.title(plot_title, size=15)
                plt.xlabel('channel')
                plt.ylabel('counts/s')
                plt.legend()
    
    if plot_display:    
        plt.show()
    else:
        return centre_positions, fwhm, amplitudes, errors

def sources_info():
    """ 
    open the text file in which all needed information about the sources is 
    (half-lives and energies) and returns it
    """
    filename = 'sources_info.txt'
    data = []
    with open(filename, 'r', errors = 'ignore') as file:
        for line in file:
            data.append(line.strip().split('\t'))

    i = 0
    while data[i] != ['SOURCES HALF-LIVES (yr):']:  # separates energies info. 
                                                    # from half-lives info.
        i += 1
    energies = data[1:i-1]
    halflife = data[i+1:]

    return energies, halflife

def calibration(detector, angle = '0', plot_display = True):
    """ 
    displays the calibration function between the channels of a given detector
    and the energy in electron-Volt (or simply return the parameters of the 
    polynomial function that models the relation)
    """
    
    energies = sources_info()[0]    # get the energies for all sources

    # this part is dedicated to obtaining the centroids of the photopeaks of 
    # all sources with the specified detector
    centre_positions = []
    photopeak_energy = []
    uncertainty = []
    for i in range(len(energies)):  # for each source
        source = energies[i][0]
        model_info = gaussian_fit(detector, source, angle, False)
        source_centre_pos = model_info[0]  # gives all cendroids for one source
        errors = model_info[3]  # gives the error on all parameters on all
                                # photopeaks
        
        for j in range(len(source_centre_pos)):
            centre_positions.append(source_centre_pos[j])  # save all centroids
                                                        # in the dedicated list
            photopeak_energy.append(float(energies[i][j+1]))
            uncertainty.append(abs(errors[j][0]))    # keep only the error on the 
                                                # centroids
    
    centre_positions = np.asarray(sorted(centre_positions))
    photopeak_energy = np.asarray(sorted(photopeak_energy))
    
    params = ('a', 'b', 'c')

    # for CdTe, as we only have two photopeaks, the so called model becomes a
    # simple line between these two points on the graph:
    if detector == 'CdTe':
        a = (photopeak_energy[2] - photopeak_energy[0]) / \
            (centre_positions[2] - centre_positions[0])
        b = photopeak_energy[0] - a * centre_positions[0]
        popt = [0, a, b]
        f_label = f'f(x) = ax + b\na = {round(a, 4)}\nb = {round(b, 4)}'
        centre_positions = [centre_positions[0], centre_positions[2]]  
                        # the modification of the list is needed, because
                        # to be able to have a better model than a line in the
                        # efficiency part, I will consider another peak that 
                        # might be a photopeak. I don't use this potential 
                        # photopeak in calibration because unlike in the 
                        # efficiency model, the expected model here is an 
                        # affine function. So I remove it from the photopeak
                        # list
        photopeak_energy = [photopeak_energy[0], photopeak_energy[2]]
        uncertainty = [uncertainty[0], uncertainty[2]]

    else:   # for the other detectors, we are fitting a polynomial function
        popt, pcov = curve_fit(polynomial, centre_positions, photopeak_energy,\
                                sigma = uncertainty, absolute_sigma = True)
        np.sqrt(np.diag(pcov))
        f_label = f'f(x) = ax² + bx + c\n {format_result(params, popt, pcov)}'
    
    if plot_display:    # we plot the obtained calibration function on our data
                        # with their uncertainties

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(centre_positions, photopeak_energy, xerr=uncertainty, \
                    fmt = '+', color = 'gray', zorder = 1)
        plt.title(f'Calibration for the {detector} detector', size = 15)
    
        print(f'\ncalibration function: {f_label}')
        
        # we create a longer list of channels, so that the calibration curve
        # looks smoother
        channels = np.asarray(np.linspace(centre_positions[0] - 1, \
                                         centre_positions[-1] + 1))
        ax.plot(channels, polynomial(channels, *popt), color='olive')
        plt.xlabel('channel')
        plt.ylabel('energy (keV)')
        plt.text(0.5, 0.1, f'\ncalibration function: {f_label}', \
                 fontsize = 10, color = "olive", transform = ax.transAxes)

        plt.show()

    else:    
        return [*popt]

def calibration_function(detector, x):
    """ 
    returns the function obtained from the calibration function
    """

    a, b, c = calibration(detector, plot_display = False)

    return a * x ** 2 + b * x + c

def parameter_extraction(detector, angle, with_barium=True, \
                         CdTe_resolution=False):
    """ 
    returns the parameters and their associated errors for all sources observed
    withy a specific detector. It also returns a list called 'repartition' 
    chich indicates the number of photopeaks of each source
    """
    # to exclude barium or not from the sources examined (usefull for
    # resolution)
    if with_barium:
        sources = ['americium','barium', 'caesium', 'cobalt']
    else:
        sources = ['americium', 'caesium', 'cobalt']

    all_centroids, all_fwhm, all_amplitudes, all_errors, repartition = [], [], [], [], []
    for element in sources:
        centroid, sigma, amplitude, errors = gaussian_fit(detector, element, angle, False, CdTe_resolution)
        i = 0
        fwhm = [w * (2 * np.sqrt(2 * np.log(2))) for w in sigma]
        for i in range(len(fwhm)):
            # the cendroids and the FWHM are added to the lists in eV (see the
            # report for the detailed explanation on how to convert the FWHM in
            # eV)
            all_centroids.append(calibration_function(detector, centroid[i]))
            all_fwhm.append(calibration_function(detector, fwhm[i]) + calibration_function(detector, 0))
            all_amplitudes.append(amplitude[i])
            all_errors.append(errors[i])
            i += 1
        repartition.append(i)

    return all_centroids, all_fwhm, all_amplitudes, all_errors, repartition

def resolution(detector, angle = '0', plot_display = True):
    """ 
    display the energy resolution as a function of energy (or only the
    parameters of the model if needed) for a specific detector
    """
    
    if detector == 'CdTe':  # for CdTe, I will use other the parameters of 
                            # the gaussian model of other peaks than only the
                            # photopeaks (or else I would only have two points)
        parameters = parameter_extraction(detector, angle, True, True)
    else:
        parameters = parameter_extraction(detector, angle, False)
    
    centroids = parameters[0]
    fwhm = parameters[1]
    errors = parameters[3]
    
    # define the resolution and its associated (propagated) error:
    resolution = [fwhm[i] / centroids[i] for i in range(len(fwhm))]
    fwhm_uncertainties = [2 * np.sqrt(2 * np.log(2)) * errors[i][1] for i in \
                          range(len(fwhm))]
    centroids_uncertainties = [errors[i][0] for i in range(len(centroids))]
    R_uncertainties = [np.sqrt(fwhm_uncertainties[i] ** 2 + \
                (fwhm[i] * centroids_uncertainties[i] / centroids[i]) ** 2) / \
                centroids[i] for i in range(len(fwhm))]

    resolution = np.asarray(resolution)
    fwhm = np.asarray(fwhm) 
    centroids = np.asarray(centroids)
    R_uncertainties = np.asarray(R_uncertainties)

    # definition of R²E² (fwhm_2), that I will use for the model:
    fwhm_2 = np.asarray([fwhm[i] ** 2 for i in range(len(centroids))])
    
    # I sort all lists by increasing associated centroid:
    sorted_indices = np.argsort(centroids)
    centroids = centroids[sorted_indices]
    fwhm = fwhm[sorted_indices]
    fwhm_2 = fwhm_2[sorted_indices]
    resolution = resolution[sorted_indices]
    R_uncertainties = R_uncertainties[sorted_indices]

    # I fit the model to the data:
    popt, pcov = curve_fit(polynomial, centroids, fwhm_2)
    np.sqrt(np.diag(pcov))

    params = ('a', 'b', 'c') 
    f_label = f'\nResolution function at {angle}° angle: f(R²) '\
                    '= a + b/E + c/E²\n' 
    if plot_display:    # plot the resolution with its uncertainty and the
                        # model, on a log-log scale:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(centroids, resolution, xerr = centroids_uncertainties, \
                     yerr = R_uncertainties, fmt = '+', \
                        label = 'resolution with uncertainty', \
                            color = 'gray', zorder = 1)
        plt.title(f'Resolution-Energy relation for the {detector} detector', \
                  size = 15)
        
        plt.text(0.01, 0.02, f'{f_label}{format_result(params, popt, pcov)}',\
             fontsize = 10, color = "olive", transform = ax.transAxes)
        
        # we create a longer list of energies, so that the resolution curve
        # looks smoother
        extended_centroids = np.linspace(centroids[0], centroids[-1])
        fwhm_2_fit = polynomial(extended_centroids, *popt)
        resolution_fit = [np.sqrt(fwhm_2_fit[i]) / extended_centroids[i] for \
                          i in range (len(fwhm_2_fit))]

        ax.plot(extended_centroids, resolution_fit, color = 'olive', \
                label = 'powerlaw fit')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('centroids of photopeaks (keV)')
        plt.ylabel('resolution')
        plt.legend()
        plt.show()

    if angle != '0':
        print(f_label, format_result(params, popt, pcov))
    else:
        print(f_label, format_result(params, popt, pcov))
        
    if not plot_display:
        return [*popt]

def compare_resolutions():
    """ 
    plots the resoltion for the three detectors on one graph to facilitate the
    comparison
    """

    fig, ax = plt.subplots(figsize=(9, 5))
    centroids = np.linspace(1, 1200)
    for detector in ('NaITl', 'BGO', 'CdTe'):
        popt = resolution(detector, plot_display = False)
        fwhm_2 = polynomial(centroids, *popt)
        resolution_fit = [np.sqrt(fwhm_2[i]) / (centroids[i]) for \
                            i in range (len(fwhm_2))]
        ax.plot(centroids, resolution_fit, label = f'{detector} detector')
    plt.title('Comparison of resolutions', size = 15)    
    plt.xlabel('centroids of photopeaks (keV)')
    plt.ylabel('resolution')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

def G_factor(detector, angle):
    """ 
    returns the solid angle of the detector, which dimensions are specified in 
    the 'detector.txt' text file, seen from the source at a certain angle from
    the axis of the detector (detailed in the report)
    """
    
    filename = 'detectors.txt'

    data = pd.read_csv(filename, sep=r'\s+', names = \
                    ["Detector", "Radius", "Length", "Distance"], skiprows = 1)
    data.set_index("Detector", inplace=True)

    d = data['Distance'][detector]
    R = data['Radius'][detector]
    L = data['Length'][detector]
    theta = float(angle) * np.pi/180    # put the angle in radian units

    G = (2 * L * R * np.sin(theta) + np.pi * R ** 2 * np.cos(theta)) / d ** 2
    
    return G

def G_factor_display(detector):
    """ 
    plots the evolution of the geometric factor with the angle
    """
    
    angles = np.linspace(0, 90)
    G = [G_factor(detector, angle) for angle in angles]
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.title(f'Evolution of the geometric factor of the {detector} detector')
    ax.plot(angles, G)
    plt.xlabel('angle (°)')
    plt.ylabel('G (str)')
    plt.show()

def activity():
    """ 
    returns the activities of all sources at the time we realised the 
    experimentation (ie around October 1st)
    """
    
    filename = 'upstairs.dat'
    data = []
    with open(filename, 'r', errors='ignore') as file:
        for line in file:
            data.append(line.strip().split(','))
    
    halflife = sources_info()[1]    
    data = data[5:9]    # keep only the sources that we studied
    date_format = " %Y-%m-%d %H:%M:%S %Z"   # format of the date given in the
                                            # 'upstairs.dat' file
    
    activities = []

    for source in data:
        source_activity = float(source[3])  # initial activity 
        ref_time = source[5]    # referency time ; then put it in seconds:
        ref_time = datetime.strptime(ref_time, date_format)
        ref_time_seconds = int(ref_time.replace( \
            tzinfo=timezone.utc).timestamp())
        # put the date of the experiment in the same format, and then put it in
        # seconds:
        lab_date = ' 2024-10-01 12:00:00 GMT'
        lab_date = datetime.strptime(lab_date, date_format)
        lab_date_seconds = int(lab_date.replace( \
            tzinfo=timezone.utc).timestamp())

        # time since reference time:
        elapsed_time = (lab_date_seconds - ref_time_seconds) / \
            (365 * 24 * 3600)
        # in all text documents, the sources are arranged alphabetically, so we
        # can use the index to refer from one file to another. 
        source_halflife = float(halflife[data.index(source)][1])

        # forumla applided to the obtained values to have the activity:
        source_activity = source_activity * \
                        np.exp(- np.log(2) * elapsed_time / source_halflife)
        activities.append(round(source_activity, 4))

    return activities

def efficiency(detector, angle = '0', plot_display = True):
    """ 
    plots the efficiency as a function of energy for a given detector (or gives
    the values of its model for further purposes)
    """
    
    # have the activities of all sources:
    activities = activity()

    parameters = parameter_extraction(detector, angle)
    count_rates = parameters[2]
    errors = parameters[3]
    centroids = np.asarray(parameters[0])
    repartition = parameters[4]
    # creation of the list of uncertainties for the centroids:
    centroids_uncertainties = np.asarray([errors[i][0] for i in \
                                          range(len(centroids))])
    # this small loop creates the list of activities for each photopeak's 
    # source. eg if there is 3 barium peaks, the barium activity will happen
    # three times in the list (one after the other).
    activities_repartition = []
    for i in range(len(repartition)):
        for j in range (repartition[i]):
            activities_repartition.append(activities[i])

    # creation of the absolue efficiency list:
    absolute_efficiency = [count_rates[i] / \
                           activities_repartition[i] for i in \
                            range(len(count_rates))]

    # creation of the intrinsic efficiency list, with the geometric factor:
    geometric_factor = G_factor(detector, angle)
    intrinsic_efficiency = np.asarray([e * geometric_factor for e \
                                       in absolute_efficiency])
    
    # list of intrinsic efficiency uncertainties:
    eff_uncertainties = np.asarray([centroids_uncertainties[i] * \
                    geometric_factor / activities_repartition[i] for i in \
                    range(len(centroids_uncertainties))])
    # creation of lists that will be used for the model:
    log_centroids = np.asarray(np.log(centroids))
    log_intrinsic_eff = np.asarray(np.log(intrinsic_efficiency))

    # sort all lists in ascending order of cendroid:
    sorted_indices = np.argsort(centroids)
    log_centroids = log_centroids[sorted_indices]
    log_intrinsic_eff = log_intrinsic_eff[sorted_indices]
    centroids = centroids[sorted_indices]
    intrinsic_efficiency = intrinsic_efficiency[sorted_indices]
    eff_uncertainties = eff_uncertainties[sorted_indices]
    centroids_uncertainties = centroids_uncertainties[sorted_indices]

    # fit the model:
    popt, pcov = curve_fit(polynomial, log_centroids, log_intrinsic_eff)
    np.sqrt(np.diag(pcov))
    
    if plot_display:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(centroids, intrinsic_efficiency, xerr = \
                centroids_uncertainties, yerr = eff_uncertainties, fmt = '+', \
                        label = 'intrinsic efficiency with uncertainty', \
                            color = 'gray', zorder = 1, capsize=5) # plot the data
        # we create a longer list of energies, so that the resolution curve
        # looks smoother:
        extended_centroids = np.linspace(centroids[0], centroids[-1])
        # convert the model associated to ln(ε) = f(ln(E) to simply ε) for the
        # plot:
        ax.plot(extended_centroids, np.exp(polynomial(np.log( \
            extended_centroids), *popt)), color='olive', label='fit')

        plt.xscale('log')
        plt.yscale('log')
        plt.title(
        f'Intrinsic efficiency-Energy relation for the {detector} detector', \
            size=15)
        plt.xlabel('centroïds of photopeaks (keV)')
        plt.ylabel('intrinsic efficiency')
        plt.legend()
        plt.show()

        params = ('a', 'b', 'c')
        if angle != '0':
            print(
f'\nEfficiency function at {angle}° angle: ln(ε) = a ln(E)² + b ln(E) + c\n', \
                format_result(params, popt, pcov))
        else:
            print(f'\nEfficiency function: ln(ε) = a ln(E)² + b ln(E) + c\n', \
                    format_result(params, popt, pcov))
                
    return [*popt]

def compare_efficiency():
    """ 
    plots the efficiency for the three detectors on one graph to facilitate the
    comparison
    """

    fig, ax = plt.subplots(figsize=(9, 5))
    centroids = np.linspace(10, 1200)
    for detector in ('NaITl', 'BGO', 'CdTe'):
        popt = efficiency(detector, plot_display = False)
        fwhm_2 = polynomial(centroids, *popt)
        efficiency_fit = np.exp(polynomial(np.log(centroids), *popt))
        ax.plot(centroids, efficiency_fit, label = f'{detector} detector')
    plt.title('Comparison of efficiencies', size = 15)    
    plt.xlabel('centroids of photopeaks (keV)')
    plt.ylabel('intrinsic efficiency')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(top=10)    # set the max of efficiency to 10 on the graph, because
                        # the efficiency value is between 0 and 1
    plt.legend()
    plt.show()

def off_axis_resolution(detector):
    """ 
    displays for a specified detector the resolution for different angles on 
    the same graph
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.title(f'Resolution for different angles for the {detector} detector')

    # longer list of energies, so that the resolution curve looks smoother:
    centroids = np.linspace(1, 1200)

    if detector == 'CdTe':  # because the data at 90° with CdTe are very bad
        angles_list = ('0', '30', '60')
    else:
        angles_list = ('0', '30', '60', '90')
    
    for angle in angles_list:
        # get the values of the resolution model parameters and plot the 
        # resolution for a each angle:
        popt = resolution(detector, angle, False)

        resolution_fit = polynomial(centroids, *popt)
        resolution_fit = [np.sqrt(resolution_fit[i] / (centroids[i]**2))  \
                          for i in range (len(resolution_fit))]
        
        ax.plot(centroids, resolution_fit, label=f'angle {angle}°')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('centroïds of photopeaks (eV)')
    plt.ylabel('resolution')
    plt.legend()
    plt.show()

def off_axis_resolution_energy(detector):
    """ 
    plots the variation of the energy resolution as a function of angle for 
    different energies (normalised by the on-axis values)
    """

    params = ('a', 'b', 'c')
    # get the parameters' values of the on-axis resolution model:
    popt_0 = resolution(detector, '0', False)

    if detector == 'CdTe':
        angles_list = ('0', '30', '60')
        angles_values = np.asarray([0, 30, 60])
        energy = [40, 50, 60]   # list of energies that will be used for CdTe,
                                # which are low energies because we only have
                                # indentified photopeaks in this region of 
                                # energies
    else:
        angles_list = ('0', '30', '60', '90')
        angles_values = np.asarray([0, 30, 60, 90])
        energy = [100, 300, 661.657]    # list of energies that will be used 
                                        # for the other detectors, that include
                                        # the energy of the Caesium photopeak
    
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.title(
f'Resolution-angle relation for the {detector} detector at {energy}keV energy')
        
    for e in energy:
        resolution_energy = []
        # resolution when source is on-axis at energy e: 
        on_axis_resolution = np.sqrt(polynomial(e, *popt_0)) / e

        for angle in angles_list:
            # have the values of the resolution for each angle at energy e
            popt = resolution(detector, angle, False)
            resolution_energy.append(np.sqrt(polynomial(e, *popt)) / e)
        
        # normalisation of the resolution: divided by the on-axis resolution:
        normalised_resolution = np.asarray([r / on_axis_resolution for r \
                                             in resolution_energy])
        ax.plot(angles_values, normalised_resolution, '+', color='dimgrey')

        # fit the curve of the angle against the normalised resolution (for 
        # each energy of the predefined list)
        popt, pcov = curve_fit(polynomial, angles_values, \
                               normalised_resolution)
        np.sqrt(np.diag(pcov))
        
        # longer list of angles, so that the resolution curve looks smoother:        
        extended_angles = np.linspace(1, angles_values[-1])
        ax.plot(extended_angles, polynomial(extended_angles, *popt), \
                label=f'polynomial fit for {e} keV')

        print(
f'\nEnergy resolution variation with angle for {e}keV: R(θ) = aθ² + bθ + c\n', \
            format_result(params, popt, pcov, 8))
    
    plt.yscale('log')
    plt.xlabel('angle (°)')
    plt.ylabel('energy resolution / on-axis resolution')
    plt.legend()
    plt.show()

def off_axis_efficiency(detector):
    """ 
    displays for a specified detector the efficiency for different angles on 
    the same graph
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.title(f'Efficiency for different angles for the {detector} detector')
    
    # longer list of energies, so that the resolution curve looks smoother:        
    centroids = np.linspace(1, 2000)
    if detector == 'CdTe':  # because the data at 90° with CdTe are very bad
        angles_list = ('0', '30', '60')
    else:
        angles_list = ('0', '30', '60', '90')

    for angle in angles_list:
        # get the values of the efficiency model parameters and plot the 
        # efficiency for a each angle:
        popt = efficiency(detector, angle, False)
        ax.plot(centroids, np.exp(polynomial(np.log(centroids), *popt)), \
                label= f'angle {angle}°')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(
        f'Intrinsic Efficiency-Energy relation for the {detector} detector', \
            size = 15)
    plt.xlabel('centroïds of photopeaks (eV)')
    plt.ylabel('intrinsic efficiency')
    plt.legend()
    plt.show()

def off_axis_efficiency_normalised(detector):
    """ 
    displays for a specified detector the normalised efficiency for different 
    angles on the same graph (same as the off_axis_efficiency function but
    normalised by the on-axis efficiency)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.title(f'Efficiency for different angles for the {detector} detector')

    centroids = np.linspace(1, 2000)

    popt = efficiency(detector, '0', False)
    on_axis_efficiency = np.exp(polynomial(np.log(centroids), *popt))
    if detector == 'CdTe':
        angles_list = ('0', '30', '60')
    else:
        angles_list = ('0', '30', '60', '90')
    for angle in angles_list:
        popt = efficiency(detector, angle, False)
        angle_efficiency = np.exp(polynomial(np.log(centroids), *popt))
        # normalisation when plotting:
        ax.plot(centroids, angle_efficiency / on_axis_efficiency, \
                label= f'angle {angle}°')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(
        f'Intrinsic Efficiency-Energy relation for the {detector} detector', \
            size = 15)
    plt.xlabel('centroïds of photopeaks (eV)')
    plt.ylabel('intrinsic efficiency/on-axis efficiency')
    plt.legend()
    plt.show()

def off_axis_efficiency_energy(detector):
    """ 
    plots the variation of the intrinsic efficiency as a function of angle for 
    different energies (normalised by the on-axis values)
    """
    params = ('a', 'b', 'c')
    popt_0 = efficiency(detector, '0', False)

    if detector == 'CdTe':
        angles_list = ('0', '30', '60')
        angles_values = np.asarray([0, 30, 60])
        energy = [40, 50, 60]
    else:
        angles_list = ('0', '30', '60', '90')
        angles_values = np.asarray([0, 30, 60, 90])
        energy = [100, 300, 661.657]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.title(
f'Efficiency-angle relation for the {detector} detector at {energy}keV energy')
        
    for e in energy:
        efficiency_energy = []
        # efficiency on-axis at energy e
        on_axis_efficiency = np.exp(polynomial(np.log(e), *popt_0))

        for angle in angles_list:
            # have the values of the efficiency for each angle at energy e
            popt = efficiency(detector, angle, False)
            efficiency_energy.append(np.exp(polynomial(np.log(e), *popt)))
        
        # normalisation of the efficieny: division by the on-axis efficiency:
        normalised_efficiency = np.asarray([eff / on_axis_efficiency for eff \
                                            in efficiency_energy])
        ax.plot(angles_values, normalised_efficiency, '+', color='dimgrey')

        # fit the curve of the angle against the normalised efficiency (for 
        # each energy of the predefined list):
        popt, pcov = curve_fit(polynomial, angles_values, normalised_efficiency)
        np.sqrt(np.diag(pcov))

        extended_angles = np.linspace(1, angles_values[-1])
        ax.plot(extended_angles, polynomial(extended_angles, *popt), \
                label=f'polynomial fit for {e} keV')

        print(f'\nIntrinsic efficiency variation with angle for',\
              f'{e}eV: ε(θ) = aθ² + bθ + c\n', \
                format_result(params, popt, pcov))

    plt.xlabel('angle (°)')
    plt.ylabel('intrinsic efficiency / on-axis efficiency')
    plt.legend()
    plt.show()

# ---------------------------- MAIN FUNCTION ----------------------------------

def main():
    args = parse_arguments()

    print(f"\nStudy of the {args['detector']} detector")

    if args['function_display'] == "Spectrum":
        print(f"\nSpectrum of {args['source']} at a {args['angle']}° angle")
        gaussian_fit(args['detector'], args['source'], args['angle'])
    
    elif args['function_display'] == "Calibration":
        print(f"\nCalibration for the {args['detector']} detector")
        calibration(args['detector'])

    elif args['function_display'] == "Resolution":
        print(f"\nResolution for the {args['detector']} detector at a "
              f"{args['angle']}° angle")
        if args['comparison'] == 'y':
            compare_resolutions()
        else:
            resolution(args['detector'], args['angle'])

    elif args['function_display'] == "Efficiency":
        print(f"\nEfficiency for the {args['detector']} detector at a "
              f"{args['angle']}° angle")
        
        if args['comparison'] == 'y':
            compare_efficiency()
        else:
            efficiency(args['detector'], args['angle'])
    
    elif args['function_display'] == "Geometric factor":
        print(f"\nGeometric factor for the {args['detector']} detector")
        G_factor_display(args['detector'])

    elif args['function_display'] == "Off-axis response":
        if args['function_choice'] == "A":
            if args['param_choice'] == 'Resolution':
                print(f"\nEnergy resolution variation of the"
                      f"{args['detector']} detector with energy for "
                      "different angles")
                off_axis_resolution(args['detector'])
            else:
                print(f"\nIntrinsic efficiency variation of the"
                      f"{args['detector']} detector with energy"
                      "for different angles")
                off_axis_efficiency_normalised(args['detector'])
        else:
            if args['param_choice'] == 'Resolution':
                print(f"\nEnergy resolution variation of the "
                      f"{args['detector']} detector with angle for different "
                      "energies")
                off_axis_resolution_energy(args['detector'])
            else:
                print(f"\nIntrinsic efficiency variation of the "
                      f"{args['detector']} detector with angle for different"
                       " energies")
                off_axis_efficiency_energy(args['detector'])

if __name__ == "__main__":
    main()
