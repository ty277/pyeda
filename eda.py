# pandas 0.20.3
# numpy 1.13.1
# matplotlib 2.0.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EDA():
    def __init__(self, printTb=True):
        '''
        exclude_list: list of variables to be excluded from EDA
        printTB: print result tables if True
        '''
        self.printTb = printTb
        self.summary = None
        self.freq = None

    def __missing_cnt(self, series):
        """
        input:
            numpy series

        output:
            number of missing observations

        """
        return sum(series.isnull())

    def __non_missing_cnt(self, series):
        """
        input:
            numpy series

        output:
            number of non missing observations
        """
        return sum(series.notnull())

    def __levels(self, series):
        """
        input:
            data: numpy series

        output:
            levels: number of unique values of each variable including missing
        """
        return len(np.unique(series))

    def __percentiles(self, series, p):
        """
        input:
            data: numpy series

        output:
            (array): percentiles of the series
        """
        return np.percentile(series[series.notnull()], p)

    def __freq_tb(self, data, v):
        """
        data: df
        v: variable name

        output: a frequency table with frequency count of each distinct value, sorted in descending order;
                cumulative frequency;
                percent of each value;
                cumulative percent of each value.
        """
        freq_tb = pd.DataFrame(data[v].value_counts(dropna=False)).rename(columns={v: 'Freq'})
        freq_tb = freq_tb.sort_values(by=['Freq'], ascending=False)
        N = float(freq_tb.Freq.sum())
        freq_tb['Cum Freq'] = freq_tb['Freq'].cumsum()
        freq_tb['Percent'] = freq_tb['Freq'] / N
        freq_tb['Cum Percent'] = freq_tb['Cum Freq'] / N
        return freq_tb

    def overall_summary(self, data):
        """
        input:
            data: pandas data frame

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
        summary = pd.DataFrame({
            'type': data.dtypes,
            'n': data.notnull().sum(axis=0),
            'nmiss': data.isnull().sum(axis=0),
            'levels': data.apply(self.__levels, 0)
        },
            columns=['type', 'n', 'nmiss', 'levels']
        )
        # ? if no missing, nmiss shows NaN?
        summary.nmiss = summary.nmiss.fillna(0)

        # for numeric variables, gives simple stats
        data = data[list(summary[summary['type'] != 'object'].index)]
        num_summary = pd.DataFrame(
            {
                'mean': data.apply(np.mean, 0),
                'min': data.apply(np.min, 0),
                'q1': data.apply(self.__percentiles, 0, p=1),
                'q25': data.apply(self.__percentiles, 0, p=25),
                'q50': data.apply(self.__percentiles, 0, p=50),
                'q75': data.apply(self.__percentiles, 0, p=75),
                'q99': data.apply(self.__percentiles, 0, p=99),
                'max': data.apply(np.max, 0)
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

        summary = summary.join(num_summary)
        summary = summary.sort_values(by=['type'])
        self.summary = summary

        if self.printTb is True:
            return summary

    ############################ Categorical Variables: Frequency tables ##############################
    def cat_freq(self, data, cat_list=None):
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
        # if list of variables provided
        if cat_list:
            lst = cat_list
        # otherwise get a list of the non numeric variables
        else:
            types = data.dtypes
            lst = list(types[types == 'object'].index)


            # output frequency table for each variable in lst
        freq = {}
        for v in lst:
            t = self.__freq_tb(data, v)
            freq[v] = t

        self.freq = freq

        if self.printTb is True:
            for v, t in freq.iteritems():
                print '\n'
                print v
                print t

    ############################ Numeric Variables: Histograms ##############################
    def num_histogram(self, data, percentile_range=None, bins=30, num_list=None):
        """
        input:
            data(dataframe)
            percentile_range (list of length 2): lower and upper bound of percentile cut-offs (e.g. [1,99])
            bins (int): number of bins for histogram
            num_list (list): list of specified variables, fault to None and produce histograms for all numeric variables

        output:
            histogram (matplotlib.pyplot.hist)

        """

        # if list of variables provided
        if num_list:
            lst = num_list
        # otherwise get a list of the non numeric variables
        else:
            types = data.dtypes
            lst = list(types[types != 'object'].index)

        for v in lst:
            var = data[data[v].notnull()][v]
            if percentile_range:
                lower = np.percentile(var, percentile_range[0])
                upper = np.percentile(var, percentile_range[1])
                var = var[(var >= lower) & (var <= upper)]  # only look at distribution within bound
                title_msg = ": {lower} to {upper} percentile".format(lower=percentile_range[0],
                                                                     upper=percentile_range[1])
            else:
                title_msg = ""

            plt.figure(figsize=(10, 6))
            plt.hist(var, bins=bins, normed=True, color="#6495ED", alpha=0.85)
            plt.title(v + ": non missing values" + title_msg)
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.show()


    # TODO:  add bar and line chart to look at univariate relationship


