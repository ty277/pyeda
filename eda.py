############################ Helper functions ##############################
def freq_tb(data, v):
    """
    data: df
    v: variable name
    
    output: a frequency table with frequency count of each distinct value, sorted in descending order;
            cumulative frequency;
            percent of each value;
            cumulative percent of each value.
    """
    import pandas as pd

    freq_tb=pd.DataFrame(data[v].value_counts(dropna=False)).rename(columns={v: 'Freq'})
    freq_tb=freq_tb.sort_values(by=['Freq'], ascending=False)
    N=float(freq_tb.Freq.sum())
    freq_tb['Cum Freq']=freq_tb['Freq'].cumsum()
    freq_tb['Percent']=freq_tb['Freq']/N
    freq_tb['Cum Percent']=freq_tb['Cum Freq']/N
    return freq_tb  

def missing_cnt(series):
    """
    input:
        numpy series

    output:
        number of missing observations

    """  
    return sum(series.isnull())

def non_missing_cnt(series):
    """
    input:
        numpy series

    output:
        number of non missing observations
    """
    return sum(series.notnull())


def levels(series):
    """
    input:
        data: numpy series

    output:
        levels: number of unique values of each variable including missing
    """
    import numpy as np    
    return len(np.unique(series))

def percentiles(series, p):
    """
    input:
        data: numpy series

    output:
        (array): percentiles of the series
    """
    import numpy as np  
    return np.percentile(series, p)


######################### A overall summary of all variables #################################
def overall_summary(data, exlcude=None):
    """
    input:
        data: pandas data frame
        exlcude (list): list of variables to be exluded from EDA

    output:
        summary (dataframe): a summary table of all variables in the data, including:
            -name
            -data type
            -n: number of non-missing observations
            -nmiss: number of missing observations
            -levels: number of unique values, including missing
            (the following are for numerical values only):
            -mean
            -min
            -q1: 1% percentile
            -q25
            -q50
            -q75
            -q99
            -max

    """
    import pandas as pd
    import numpy as np

    if exlcude!=None:
        dat = data.drop([exlcude],1)
    else:
        dat = data

    summary = pd.DataFrame({
    'type':dat.dtypes,
    'n': dat.apply(non_missing_cnt, 0),    
    'nmiss ': dat.apply(missing_cnt, 0),
    'levels': dat.apply(levels, 0)
        },
    columns=['type','n','nmiss','levels']
        )
    #? if no missing, nmiss shows NaN?
    summary.nmiss=summary.nmiss.fillna(0)

    #for numeric variables, gives simple stats
    dat = dat[list(summary[summary['type']!='object'].index)]
    num_summary= pd.DataFrame(
        {
        'mean': dat.apply(np.mean, 0),
        'min': dat.apply(np.min, 0),
        'q1':  dat.apply(percentiles, 0, p=1),
        'q25': dat.apply(percentiles, 0, p=25),
        'q50': dat.apply(percentiles, 0, p=50),
        'q75': dat.apply(percentiles, 0, p=75),
        'q99': dat.apply(percentiles, 0, p=99),
        'max': dat.apply(np.max, 0)
        },
        columns=[
        'mean',
        'min', 
        'q1',  
        'q25', 
        'q50', 
        'q75', 
        'q99', 
        'max'])

    summary=summary.join(num_summary)
    summary=summary.sort_values(by=['type'])


    return summary



############################ Categorical Variables: Frequency tables ##############################
def cat_freq(data, cat_list=None, printout=True):
    """
    input:
        data: pandas dataframe
        cat_list(list): a list of specified variables, default None and produce frequency tables for all variables with dtype='object'
        printout (boolean): whether or not to print out the frequency tables, default to yes

    output:
        freq (dict): a dict of frequency tables of all categorical variables as values, variable names as keys
        *frequency table (dataframe): 
            -frequency count of each distinct value, sorted in descending order;
            -cumulative frequency;
            -percent of each value;
            -cumulative percent of each value.


    """
    import pandas as pd
    import numpy as np

    #if list of variables provided
    if cat_list:
        lst=cat_list
    #otherwise get a list of the non numeric variables
    else:   
        types=data.dtypes
        lst = list(types[types=='object'].index)        


    #output frequency table for each variable in lst
    freq={}
    for v in lst:
        t=freq_tb(data, v)
        freq[v]=t

    if printout==True:
        for v, t in freq.iteritems():
            print '\n'
            print v
            print t

    return freq




############################ Numeric Variables: Histograms ##############################
def num_histogram(data, percentile_range=None, bins=30, num_list=None):
    """
    input:
        data(dataframe)
        percentile_range (list of length 2): lower and upper bound of percentile cut-offs (e.g. [1,99])
        bins (int): number of bins for histogram
        num_list (list): list of specified variables, fault to None and produce histograms for all numeric variables

    output:
        histogram (matplotlib.pyplot.hist)

    """
    import matplotlib.pyplot as plt
    import numpy as np

    #if list of variables provided
    if num_list:
        lst=num_list
    #otherwise get a list of the non numeric variables
    else:   
        types=data.dtypes
        lst = list(types[types!='object'].index)      

    for v in lst:
        var = data[v]
        if percentile_range:
            lower = np.percentile(var, percentile_range[0])
            upper = np.percentile(var, percentile_range[1])
            var = var[(var >= lower) & (var <= upper)]
            title_msg = ": {lower} to {upper} percentile".format(lower=percentile_range[0], upper=percentile_range[1])
        else:
            title_msg = ""

        plt.figure(figsize=(10,6))
        plt.hist(var, bins=bins, normed=True, color="#6495ED", alpha=0.85)
        plt.title(v+title_msg)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()




