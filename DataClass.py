import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import xlrd
import os
import warnings
import numpy as np
from scipy import stats

import datetime
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from side_functions import *


class MSData(object):
    def __init__(self, filename=None, filetype=None, instrument=None):
        """
        object holding LA-ICP-MS data for data reduction
        :param filename: str name of the file of measured MS data
        `:param filetype: str type of the file ['csv', 'xlsx', 'asc']
        :param instrument: str type of MS instrument used ['Element', 'Agilent']
        """
        if filename:
            if instrument == 'Element':
                skipfooter = 4
                header = 1
                drop = 9
            elif instrument == 'Agilent':
                skipfooter = 4
                header = 3
                drop = 3
            else:
                skipfooter = 0
                header = 0
                drop = 0

            if filetype == 'xlsx':
                pwd = os.getcwd()
                os.chdir(os.path.dirname(filename))
                self.imported = pd.ExcelFile(filename)
                self.data = self.imported.parse(
                    0, index_col=0, skipfooter=skipfooter, header=header)
                self.data = self.data.drop(self.data.index[:drop], axis=0)
                os.chdir(pwd)
                # TODO xlsx doesnt work with agilent type
            elif filetype == 'csv':
                pwd = os.getcwd()
                os.chdir(os.path.dirname(filename))
                self.data = pd.read_csv(filename, sep=',', index_col=0, skipfooter=skipfooter,
                                        header=header, engine='python')
                os.chdir(pwd)
            elif filetype == 'asc':
                pwd = os.getcwd()
                os.chdir(os.path.dirname(filename))
                self.data = pd.read_csv(filename, sep='\t', index_col=0, skipfooter=skipfooter,
                                        header=header, engine='python')
                self.data = self.data.drop(self.data.index[:drop], axis=0)
                self.data.dropna(axis=1, how='all', inplace=True)
                self.data = self.data.apply(pd.to_numeric, errors='coerce')
                os.chdir(pwd)
            else:
                warnings.warn('File type not supported.')

            self.data.index = self.data.index.astype('float32')
            self.time = self.data.index
            self.elements = list(map(elem_resolution, self.data.columns))
            self.data.columns = self.elements

        self.srms = pd.ExcelFile('./SRM.xlsx').parse(index_col=0)
        self.sum_koeficients = pd.ExcelFile(
            './default_sum_koef.xlsx').parse(0, index_col=0, header=None).to_dict()[1]

        self.srm = None
        self.iolite = None
        self.names = None
        self.internal_std = None
        self.ablation_time = None

        self.laser_off = []
        self.laser_on = []
        self.skip = {'bcg_start': 0,
                     'bcg_end': 0,
                     'sample_start': 0,
                     'sample_end': 0}    # time in seconds to skip from each bcg and sample

        self.filter_line = None
        self.starts = None
        self.ends = None
        self.bcg = None
        self.average_peaks = None
        self.ratio = None
        self.quantified = None
        self.lod = None
        self.correction_elements = None
        self.corrected_IS = None
        self.corrected_SO = None

        self.dx = None
        self.dy = None
        self.maps = {}
        self.qmaps = {}

        self.regression_values = {}
        self.regression_equations = {}

    def read_param(self, path):
        xl = pd.ExcelFile(path)
        if 'names' in xl.sheet_names:
            self.names = list(xl.parse('names', header=None)[0])
        if 'internal standard' in xl.sheet_names:
            self.internal_std = xl.parse(
                'internal standard', index_col=0, header=0)
        if 'total sum' in xl.sheet_names:
            self.sum_koeficients = xl.parse(
                'total sum', index_col=0, header=None).to_dict()[1]

    def read_iolite(self, path):
        pwd = os.getcwd()
        os.chdir(os.path.dirname(path))
        self.iolite = pd.read_csv(path, sep=",", engine='python')
        os.chdir(pwd)
        print(names_from_iolite(self.iolite))

    def plot_data(self, ax=None):
        if ax is None:
            fig, ax = plt.subplot()
        ax.plot(self.data[self.laser_on, :])
        ax.show()

    def set_filtering_element(self, element):
        if element == 'sum':
            self.filter_line = self.data.sum(1)
        elif element in self.elements:
            self.filter_line = self.data[element]
        else:
            warnings.warn(
                'Element selected for filtering laser ON not in measured elements. Falling back to sum.')
            self.filter_line = self.data.sum(1)

    def set_ablation_time(self, time):
        # set time of ablation spot/line in seconds
        self.ablation_time = time

    def set_skip(self, bcg_s=None, bcg_e=None, sig_s=None, sig_e=None):
        # set time skipped on start and end of background and ablation in seconds
        if bcg_s is not None:
            self.skip['bcg_start'] = bcg_s
        if bcg_e is not None:
            self.skip['bcg_end'] = bcg_e
        if sig_s is not None:
            self.skip['sample_start'] = sig_s
        if sig_e is not None:
            self.skip['sample_end'] = sig_e

    def time_to_number(self, time):
        """
        takes time in seconds returns number of measured values
        depends on integration time of MS method
        """
        val = len(self.time[0:(
            np.abs(np.array(self.time.values, dtype=np.float32)-np.abs(time))).argmin()])
        if time < 0:
            val = -val
        return val

    def create_selector_iolite(self, start):
        # select starts and ends of ablation using iolite file
        if self.iolite.empty:
            print('Warning: Iolite not created.')
            return
        lst = [x for x in self.iolite.loc[:6,
                                          ' Comment'] if isinstance(x, str)]

        if len(lst) == 2:
            print('>>> Selecting spots.')
            difflst = get_diff_lst(self.iolite)
        elif len(lst) == 1:
            print('>>> Selecting lines.')
            difflst = get_diff_lst_line(self.iolite)
        else:
            print('Warning: Iolite not selected.')
        timeindex = []
        for i in range(0, len(difflst)+1):
            timeindex.append(sum(difflst[:i])+start)
        index = [get_index(self.data, x) for x in timeindex]

        self.starts = [index[i] for i in range(len(index)) if i % 2 == 0]
        self.ends = [index[i] for i in range(len(index)) if i % 2 != 0]

        self.create_on_off()

    def create_selector_bcg(self, bcg_sd, bcg_time):
        """
        select starts and ends of ablation based on selected element or sum of all using treshold
        calculated from background
        """
        bcg_nr = self.time_to_number(bcg_time)
        bcg = self.filter_line.iloc[0:bcg_nr].mean()
        std = self.filter_line.iloc[0:bcg_nr].std()
        ind = [True if value > bcg+bcg_sd *
               std else False for value in self.filter_line]
        ind2 = ind[1:]
        ind2.append(False)
        index = [i for i in range(0, len(ind)) if ind[i] != ind2[i]]

        self.starts = [index[i] for i in range(len(index)) if i % 2 == 0]
        self.ends = [index[i] for i in range(len(index)) if i % 2 != 0]

        self.create_on_off()

    def create_selector_gradient(self, time_of_cycle=100):
        """
        selects starts and ends of ablation based on selected element or sum of all using gradient
        param time_of_cycle: time of the ablation and half of the pause between ablations in seconds 
        """
        n = self.time_to_number(
            time_of_cycle)  # number of values for one spot and half bcg
        self.ends = list(argrelextrema(np.gradient(
            self.filter_line.values), np.greater_equal, order=n)[0])
        self.starts = list(argrelextrema(np.gradient(
            self.filter_line.values), np.less_equal, order=n)[0])
        print(self.starts)
        print(self.ends)
        self.create_on_off()

    def create_on_off(self):
        """
        from starts and ends of ablation create laser_on and laser_off with skipped values
        """
        self.laser_off = []
        self.laser_on = []

        self.laser_off.append(
            (0+self.time_to_number(self.skip['bcg_start']), self.starts[0]-self.time_to_number(self.skip['bcg_end'])))

        for i in range(len(self.starts)-1):
            self.laser_off.append((self.ends[i]+self.time_to_number(
                self.skip['bcg_start']), self.starts[i+1]-self.time_to_number(self.skip['bcg_end'])))
            self.laser_on.append((self.starts[i]+self.time_to_number(
                self.skip['sample_start']), self.ends[i]-self.time_to_number(self.skip['sample_end'])))

        self.laser_off.append((self.ends[-1]+self.time_to_number(self.skip['bcg_start']), len(
            self.time)-2-self.time_to_number(self.skip['bcg_end'])))
        self.laser_on.append((self.starts[-1]+self.time_to_number(
            self.skip['sample_start']), self.ends[-1]-self.time_to_number(self.skip['sample_end'])))

    def graph(self, ax=None, logax=False, el=None):
        """
        create matplotlib graph of intensity in time for ablation
        highlights ablation part and background signal
        """
        if ax == None:
            fig, ax = plt.subplots()

        ax.cla()
        ax.clear()
        # if element is defined, plot only one element, otherwise all
        if el:
            self.data.plot(ax=ax, y=el, kind='line', legend=False)
        else:
            self.data.plot(ax=ax, kind='line', legend=False)

        if logax:
            ax.set_yscale('log')

        if self.starts and self.ends:
            # create lines for start and end of each ablation
            for i in range(0, len(self.starts)):
                ax.axvline(x=self.time[self.starts[i]],
                           color='blue', linewidth=2)
            for i in range(0, len(self.ends)):
                ax.axvline(x=self.time[self.ends[i]],
                           color='blue', linewidth=2)

        if self.laser_off:
            # higlights bacground
            for off in self.laser_off:
                #print(self.time[off[0]], self.time[off[1]])
                try:
                    ax.axvspan(
                        self.time[off[0]], self.time[off[1]], alpha=0.2, color='red')
                except:
                    warnings.warn('something is wrong')

        if self.laser_on:
            # higlihts ablation
            for on in self.laser_on:
                ax.axvspan(self.time[on[0]], self.time[on[1]],
                           alpha=0.2, color='green')

        plt.show()

    def set_srm(self, srm):
        # select reference material used for quantification
        if isinstance(srm, list):
            self.srm = self.srms.loc[srm, :]
            return
        if srm in self.srms.index:
            self.srm = self.srms.loc[srm, :]

    def background(self, elem, scale):
        if scale == 'beginning':
            line = list(self.data[elem])
            self.bcg = line[:self.laser_off[0][1]]
            print(self.bcg)
            # self.bcg = sum(self.bcg)/len(self.bcg)
            # print(self.bcg)

    def setxy(self, dx, dy):
        # set x and y distance for elemental map
        self.dx = dx
        self.dy = dy

    def integrated_area(self, elem):
        # calculate area of a spot for given element
        # returns list of areas
        areas = []
        line = self.data[elem]

        if not self.laser_on and not self.laser_off:
            print('Warning')
            return
        for i in range(0, len(self.laser_on)):
            on = self.laser_on[i]
            off_before = self.laser_off[i]
            off_after = self.laser_off[i+1]

            sample_y = list(line)[on[0]:on[1]]
            sample_x = list(line.index)[on[0]:on[1]]

            bcg_y = list(line)[off_before[0]:off_before[1]] + \
                list(line)[off_after[0]:off_after[1]]
            bcg_x = list(line.index)[
                off_before[0]:off_before[1]] + list(line.index)[off_after[0]:off_after[1]]

            gradient, intercept, r_value, p_value, std_err = stats.linregress(
                bcg_x, bcg_y)
            new_y_sample = [gradient*x+intercept for x in sample_x]
            areas.append(np.trapz(sample_y, sample_x) -
                         np.trapz(new_y_sample, sample_x))
        return areas

    def mean_intensity(self, elem, scale):
        # calculate mean intensity of a spot for given element
        # returns list of means
        means = []
        line = self.data[elem]
        if not self.laser_on and not self.laser_off:
            print('Warning')
            return
        if scale == 'beginning':
            self.background(elem, scale)
            bcg_y = self.bcg

        for i in range(0, len(self.laser_on)):
            on = self.laser_on[i]
            off_before = self.laser_off[i]
            off_after = self.laser_off[i+1]

            sample_y = list(line)[on[0]:on[1]]
            if scale == 'all':
                bcg_y = list(line)[off_before[0]:off_before[1]] + \
                    list(line)[off_after[0]:off_after[1]]

            means.append(np.mean(outlier_detection(sample_y)) -
                         np.mean(outlier_detection(bcg_y)))
        return means

    def average(self, method='area', scale='all'):
        # calculate average signal for each spot with substracted background
        # method: 'area' uses integration of the peak 'intensity' uses mean of intensities

        self.average_peaks = pd.DataFrame(columns=list(self.elements))
        for elem in self.elements:
            if method == 'area':
                self.average_peaks[elem] = (self.integrated_area(elem))
            if method == 'intensity':
                self.average_peaks[elem] = (self.mean_intensity(elem, scale))

        if self.names:
            try:
                self.average_peaks.index = self.names
            except ValueError as e:
                warnings.warn('Unable to match peak names to data.')
                print(e)
        else:
            self.names = ['peak_{}'.format(i) for i in range(
                1, len(self.average_peaks.index)+1)]
            self.average_peaks.index = self.names

    def quantification(self):
        # calculate quantification of intensities or areas using selected reference material
        if not self.names or self.srm.empty or self.average_peaks.empty:
            warnings.warn('Missing data.')
        spots = self.average_peaks.iloc[[
            i for i, val in enumerate(self.names) if val != self.srm.name]]
        print(spots)
        stdsig = self.average_peaks.iloc[[i for i, val in enumerate(
            self.names) if val == self.srm.name]].mean(axis=0)
        print(stdsig)
        self.ratio = [float(self.srm[element_strip(el)]) /
                      float(stdsig[el]) for el in stdsig.index]
        print(self.ratio)
        self.quantified = spots.mul(self.ratio, axis='columns')

    def detection_limit(self, method='area', scale='all'):
        """
        calculate limit of detection for analysis
        param: method = ['area','intensity'] use same method as for the average
        param: scale = ['beginning', 'all']
        """
        if scale == 'all':
            bcg = pd.DataFrame(columns=self.data.columns)
            for (s, e) in self.laser_off:
                bcg = pd.concat([bcg, self.data.iloc[np.r_[s:e], :]])
        elif scale == 'beginning':
            bcg = self.data.iloc[self.laser_off[0][0]:self.laser_off[0][1]]
        if method == 'area':
            self.lod = (bcg.std()*self.ablation_time).mul(self.ratio)
        elif method == 'intensity':
            self.lod = (bcg.std()*3).mul(self.ratio)
        self.lod.name = 'LoD'

    def internal_standard_correction(self):
        # calculates correction for each element given in internal standard correction from PARAM file
        print(self.internal_std.columns)
        self.corrected_IS = []
        if self.internal_std.empty:
            return
        self.correction_elements = list(self.internal_std.columns)
        print(self.correction_elements)
        for el in self.correction_elements:
            print(el)
            if el in self.elements:
                self.corrected_IS.append(correction(
                    self.quantified, el, self.internal_std))
            # else:
                # self.correction_elements.remove(el)

    def total_sum_correction(self, suma=1000000):
        # calculates total sum correction using coefficients given in PARAM file
        if not self.sum_koeficients:
            warnings.warn('Missing coeficients for total sum correction.')
            return
        print(self.sum_koeficients)
        self.corrected_SO = self.quantified.copy()
        for key in self.sum_koeficients:
            elem = element_formater(key, self.corrected_SO.columns)
            if not elem:
                continue
            self.corrected_SO[elem] = self.corrected_SO[elem] / \
                self.sum_koeficients[key] * 100
        koef = suma/self.corrected_SO.sum(1)
        self.corrected_SO = self.corrected_SO.mul(list(koef), axis='rows')
        for key in self.sum_koeficients:
            elem = element_formater(key, self.corrected_SO.columns)
            if not elem:
                continue
            self.corrected_SO[elem] = self.corrected_SO[elem] * \
                self.sum_koeficients[key] / 100

    def report(self):
        if self.corrected_SO is not None:
            self.corrected_SO = self.corrected_SO.append(self.lod)
            for column in self.corrected_SO:
                self.corrected_SO[column] = [
                    round_me(value, self.lod, column) for value in self.corrected_SO[column]]

        if self.corrected_IS is not None:
            self.corrected_IS = [df.append(self.lod)
                                 for df in self.corrected_IS]
            for df in self.corrected_IS:
                for column in df:
                    df[column] = [round_me(value, self.lod, column)
                                  for value in df[column]]

        if self.quantified is not None:
            self.quantified = self.quantified.append(self.lod)
            for column in self.quantified:
                self.quantified[column] = [
                    round_me(value, self.lod, column) for value in self.quantified[column]]

    def save(self, path, data=None):
        if data is None and self.quantified is not None:
            data = self.quantified
        elif data is None and self.quantified is None:
            warnings.warn('No data to save.')

        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        if isinstance(data, list):
            for item, e in zip(self.corrected_IS, self.correction_elements):
                item.to_excel(writer, sheet_name='Normalised_{}'.format(e))
        else:
            data.to_excel(writer, sheet_name='report')
        writer.save()

    def matrix_from_time(self, elem, bcg):
        # create elemental map from time resolved LA-ICP-MS data
        if self.dx is None or self.dy is None:
            print('Warning: Missing values dx or dy.')
            return
        line = self.data[elem]
        d = {}
        tmpy = 0
        if bcg == 'beginning':
            bcg_lst = list(line)[self.laser_off[0][0]:self.laser_off[0][1]]
            print(bcg_lst)

        for i in range(0, len(self.laser_on)):
            on = self.laser_on[i]
            off_before = self.laser_off[i]
            off_after = self.laser_off[i + 1]

            tmpy = tmpy + self.dy
            if bcg == 'beginning':
                print('using beginning')
            elif bcg == 'all':
                bcg_lst = list(line)[off_before[0]:off_before[1]] + \
                    list(line)[off_after[0]:off_after[1]]
                print(bcg_lst)
            else:
                print('Warning: not a valid background method')
            arr = np.array(line)[on[0]:on[1]] - np.mean(bcg_lst)
            arr[arr < 0] = 0
            d[tmpy] = arr

        df = pd.DataFrame.from_dict(d, orient='index')
        tmpx = range(self.dx, self.dx * len(df.columns) + self.dx, self.dx)
        df.columns = tmpx
        return df

    def create_all_maps(self, bcg=None):
        for el in self.elements:
            self.maps[el] = self.matrix_from_time(el, bcg)

    def rotate_map(self, elem):
        if elem in self.maps.keys():
            rotated = np.rot90(self.maps[elem])
            if rotated.shape[0] == len(self.maps[elem].index):
                indexes = self.maps[elem].index
            else:
                indexes = self.maps[elem].columns

            if rotated.shape[1] == len(self.maps[elem].columns):
                columns = self.maps[elem].columns
            else:
                columns = self.maps[elem].index

            self.maps[elem] = pd.DataFrame(
                rotated, columns=columns, index=indexes)
        else:
            print('Warning: Matrix does not exists.')

    def elemental_image(self, elem, fig=None, ax=None, vmin=None, vmax=None, clb=True, axis=True,
                        colourmap='jet', interpolate='none', title='', units='', quantified=False, *args, **kwargs):

        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.cla()

        if quantified is True:
            if elem in self.qmaps.keys():
                data = self.qmaps[elem]
            else:
                warnings.warn('Elemental map not quantified.')
                data = self.qmaps[elem]
        else:
            if elem in self.maps.keys():
                data = self.maps[elem]
            else:
                warnings.warn('Elemental map not generated.')
                data = self.qmaps[elem]

        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=colourmap, interpolation=interpolate,
                       extent=[0, self.maps[elem].columns[-1], self.maps[elem].index[-1], 0], *args, **kwargs)  # .values
        if not axis:
            ax.axis('off')
        if clb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            clb = fig.colorbar(im, cax=cax)
            clb.ax.set_title(units)
        fig.suptitle(title)
        # plt.show()

    def quantify_map(self, elem, intercept, slope):
        if elem not in self.elements:
            warnings.warn('Element map doesnt exist.')
            return
        self.qmaps[elem] = (self.maps[elem]-intercept)/slope

    def quantify_all_maps(self):
        for elem in self.elements:
            self.quantify_map(
                elem=elem, intercept=self.regression_equations[elem][0], slope=self.regression_equations[elem][1])

    def export_matrices(self, path, quantified=False):
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        if quantified:
            for el, mapa in self.qmaps.items():
                mapa.to_excel(writer, sheet_name=el)
            writer.save()

        else:
            for el, mapa in self.maps.items():
                mapa.to_excel(writer, sheet_name=el)
            writer.save()

    def import_matrices(self, path):
        file = pd.ExcelFile(path)
        for el in file.sheet_names:
            self.maps[el] = file.parse(el, index_col=0)
        self.elements = file.sheet_names

    def get_regression_values(self, method, srm):
        self.average(method=method)
        self.set_srm(srm=srm)
        for elem in self.elements:
            self.regression_values[elem] = pd.DataFrame(
                {'x': self.srm[element_strip(elem)].values, 'y': self.average_peaks[elem].values})

    def calibration_equations(self, intercept=False):
        for elem in self.elements:
            model = LinearRegression(fit_intercept=intercept, normalize=False)
            x = np.array(self.regression_values[elem]['x']).reshape((-1, 1))
            y = np.array(self.regression_values[elem]['y'])
            model.fit(x, y)
            self.regression_equations[elem] = (
                model.intercept_, model.coef_[0])
            # print(self.regression_equations[elem])

    def calibration_graph(self, elem, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.cla()
        x = np.array(self.regression_values[elem]['x'])
        y = np.array(self.regression_values[elem]['y'])
        ax.plot(x, y, 'bo')
        # add regression line
        x = np.linspace(0, x.max(), 100)
        a = self.regression_equations[elem][1]
        b = self.regression_equations[elem][0]
        y = a * x + b
        ax.plot(x, y, '-r', label='y={:.2E}x+{:.2E}'.format(a, b))
        ax.legend(loc='upper left')
        ax.set_xlabel('concentration')
        ax.set_ylabel('intensity')
        ax.set_xlim(0,)
        ax.set_ylim(0,)
        ax.set_title(elem)
        plt.show()


if __name__ == '__main__':

    def test_spot():
        d = MSData('/Users/nikadilli/OneDrive - MUNI/Geologie/granaty_copjakova/190715/Uhr17_prd_D_LR.asc',
                   # 'C:\\Users\\Admin\\OneDrive - MUNI\\Archeologie\\Tomková\\meranie\\190407\\an2.csv',
                   # 'C:\\Users\\Veronika\\OneDrive - MUNI\\Geologie\\granaty_copjakova\\190520\\SP1b_cpx_LR.asc',
                   filetype='asc',
                   instrument='Element')
        print(d.data)

        # d.read_iolite('C:\\Users\\Veronika\\OneDrive - MUNI\\Geologie\\granaty_copjakova\\190520\\SP1b_cpx.Iolite.csv')
        d.read_param(
            '/Users/nikadilli/OneDrive - MUNI/Geologie/granaty_copjakova/190715/PARAM_Uhr17_prdA.xlsx')
        d.set_filtering_element('sum')
        d.set_srm('NIST610')
        d.set_ablation_time(60)
        d.set_skip(10, 10, 10, 15)

        # d.create_selector_iolite(90)
        # d.create_selector_gradient(120)
        d.create_selector_bcg(300, 50)

        d.graph()
        d.average(method='area')

        d.quantification()
        d.detection_limit(method='area', scale='begining')
        print(d.quantified)
        print(d.lod)
        print(d.quantified.mean())
        d.internal_standard_correction()
        # d.total_sum_correction()
        print(d.corrected_IS)
        d.report(method='correction IS')
        print(d.corrected_IS)
        # d.corrected_SO.to_excel( 'C:\\Users\\Admin\\OneDrive - MUNI\\Archeologie\\Tomková\\meranie\\190408\\an2.xlsx')
        d.save(path='/Users/nikadilli/OneDrive - MUNI/Geologie/granaty_copjakova/190715/Uhr17_prdD_area.xlsx')

    def test_map():
        d = MSData('/Users/nikadilli/Downloads/mapa vzorek 12.csv',
                   filetype='csv', instrument='raw')
        # '/Users/nikadilli/code/Ilaps/test_data/mapa1c.csv', filetype='csv', instrument='Agilent') ⁩ ▸ ⁨⁩
        d.read_iolite(
            '//Users/nikadilli/Downloads/190729vyzorek12n.Iolite.csv')
        d.set_filtering_element('sum')
        d.set_skip(3, 3, 0, 0)
        d.create_selector_iolite(20)
        d.graph()
        d.setxy(50, 50)
        d.create_all_maps(bcg='begining')
        # d.export_matrices('/Users/nikadilli/Google Drive/glioma tvorba matic/05082019/st zbytek/matica.xlsx')
        d.elemental_image(elem='Fe56', interpolate='bicubic')

    def test_calib():
        d = MSData('/Users/nikadilli/code/Ilaps/test_data/data.csv',
                   filetype='csv', instrument='raw')
        d.set_filtering_element('sum')
        d.create_selector_bcg(100, 20)
        # d.graph()
        d.get_regression_values('intensity', ['NIST612', 'NIST610'])
        d.calibration_equations()
        d.calibration_graph('Na23')

    test_map()
