# pandas 0.20.3
# numpy 1.13.1
# matplotlib 2.0.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-muted')


class EDA():
    def __init__(self, printTb=True):
        '''
        exclude_list: list of variables to be excluded from EDA
        printTB: print result tables if True
        '''
        self.printTb = printTb
        self.summary = None
        self.freq = None
        # parameters for plotting
        self.mycolors = ['royalblue', 'firebrick', 'forestgreen', 'orange', 'grey',
                         'darkturquoise', 'palevioletred', 'olive', 'sandybrown', 'mediumpurple']
        self.opacity = 0.75

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
        return len(series.unique())

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
            for v, t in iter(freq.items()):
                print('\n')
                print(v)
                print(t)

    ############################ Numeric Variables: Histograms ##############################
    def num_histogram(self, data, percentile_range=None, bins=30, width=10, height=6, num_list=None):
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

            plt.figure(figsize=(width, height))
            plt.hist(var, bins=bins, normed=True, color=self.mycolors[0], alpha=self.opacity)
            plt.title(v + ": non missing values" + title_msg)
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.show()

    def plot_bar_over_buket(self, plot_df, varname, bktname, title=None, xlabel=None, ylabel=None,
                            cl=None, figwidth=14, figheight=6,
                            bar_width=.25, xticksrotation=90):
        '''
        plot_df: data frame for plot
        varname: column name of the variable to be ploted over different buckets
        bktname: bucket column name
        '''
        n_groups = len(plot_df)
        if cl is None:
            cl = self.mycolors[0]
        if title is None:
            title = varname + ' by ' + bktname
        if xlabel is None:
            xlabel = bktname
        if ylabel is None:
            ylabel = varname

        # create plot
        fig, ax = plt.subplots()
        fig.set_figwidth(figwidth)
        fig.set_figheight(figheight)
        index = np.arange(n_groups)

        plt.bar(index + bar_width, plot_df[varname], width=bar_width, label=varname, color=cl, alpha=self.opacity)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(index + bar_width, plot_df[bktname], rotation=xticksrotation)
        plt.legend(loc=2)

        plt.tight_layout()
        plt.show()

    def plot_bar_line_over_bkt(self, plot_df, barvar, linevar, bktvar,
                               barvar2=None, linevar2=None,
                               title=None, xlabel=None, ylabel1=None, ylabel2=None,
                               width=14, height=6, bar_width=0.25):
        '''
        barvar: column name of the variable to be ploted in bars over different buckets
        linevar: column name of the variable to be ploted as line over different buckets
        bktname: bucket column name
        '''
        if title is None:
            title = ', '.join([barvar, linevar]) + ' by ' + bktvar
        if xlabel is None:
            xlabel = bktvar
        if ylabel1 is None:
            ylabel1 = barvar
        if ylabel2 is None:
            ylabel2 = linevar
        # bar chart
        n_groups = len(plot_df)

        # create plot
        fig, ax = plt.subplots()
        fig.set_figwidth(width)
        fig.set_figheight(height)
        index = np.arange(n_groups)
        ax.bar(index - bar_width / 2., plot_df[barvar], bar_width,
               alpha=self.opacity,
               color=self.mycolors[0],
               label=barvar)
        if barvar2 is not None:
            ax.bar(index + bar_width / 2., plot_df[barvar2], bar_width,
                   alpha=self.opacity,
                   color=self.mycolors[1],
                   label=barvar2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel1)
        plt.title(title)
        xtkoffset = -bar_width / 2. if (barvar2 is None) else 0
        plt.xticks(index + xtkoffset, plot_df[bktvar], rotation=90)
        plt.legend(loc='upper left')

        ax2 = ax.twinx()
        # if not comparing two pairs of values, use a different color for line, if yes, use the same color
        cidx = 1 if (barvar2 is None) else 0
        ax2.plot(index + xtkoffset, plot_df[linevar], color=self.mycolors[cidx], label=linevar,
                 linestyle='--', marker='o')
        if linevar2 is not None:
            ax2.plot(index + xtkoffset, plot_df[linevar2], color=self.mycolors[1], label=linevar2,
                     linestyle='--', marker='o')
        ax2.set_ylabel(ylabel2)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()


