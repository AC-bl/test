In the python file (called 'fichier_source.py'), choosing the detector, you can plot:
- Spectrum
- Calibration
- Resolution
- Efficiency
- Geometric factor
- Off-axis response for efficiency of resolution

Depending on which plot you choose, other parameters will be asked.

For the code to work, the data files should be named as follows:
source_angle_detector.extention

Four text files are needed:
- upstairs.txt for the activity
- photopeak_intervals.txt to have the intervals of the photopeak for each source with each detector
- detectors.txt which contains the dimensions of the three detectors
- sources_info.txt that gives the emission energy of interest (that can be identified to photopeaks) for each sources 

Explanation on the photopeak_intervals.txt construction:

For each source line:
- the first number corresponds to the max channel at which the plot will display the spectrum, which allows to see the photopeaks more clearly on the graphs.
- the Following Numbers are to read two by two: they correspond to the channel min and channel max of the photopeaks intervals. They are arranged by corresponding emission fraction


 
