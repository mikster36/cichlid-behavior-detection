import numpy
import pandas
from evaluate import change_config

"""
    Gets the average length of a cichlid from a trial based on labeled annotations
    
    Parameters
    frames: a list of String paths to .csv files of annotated data
    
    Returns
    A float representing the average length of a cichlid
    
"""
def avg_cichlid_length(frames):
    for frame in frames:
        labels = pandas.read_csv(frame)