import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

# ------------ Helper Class for Cycling Behavior Extraction and Evaluation ------------

class CycleHelper():
    def __init__(self, key_timestamp, key_consumption):
        '''
            Provides a class for evaluating the cycling behavior of heat pumps.
            Args: 
                key_timestamp: keyword for timestamp 
                key_consumption: keyword for consumption column (energy in kWh)
        '''

        assert isinstance(key_timestamp, str), 'CycleHelper: Parameter key_timestamp must be a string representation of the column name that encodes the timestamp of data frames to be processed.'
        assert isinstance(key_consumption, str), 'CycleHelper: Parameter key_conumption must be a string representation of the column name that encodes the kWh-consumption of data frames to be processed.'
        
        self.key_timestamp = key_timestamp
        self.key_consumption = key_consumption
    
    def __check_key_columns__(self, df): 
        '''
            Checks if assigned key_timestamps and key_consumption are columns in a given data frame
            Args: 
                df: data frame to be checked
            Returns: 
                True if keys exist 
        '''
        if self.key_consumption not in df.columns.values: 
            return False
        if self.key_timestamp not in df.columns.values: 
            return False

    def set_key_timestamp(self, key_timestamp): 
        '''
            Sets parameter key_time_stamp. 
            Args: 
                key_time_stamp: string to define the key_time_stamp parameter 
        '''
        assert isinstance(key_timestamp, str) or key_timestamp is None, 'CycleHelper: Parameter key_timestamp must be a string representation of the column name that encodes the timestamp of data frames to be processed or None.'
        self.key_timestamp = key_timestamp

    def get_key_timestamp(self): 
        '''
            Returns string of current key_timestamp parameter. 
        '''
        return self.key_timestamp

    def set_key_consumption(self, key_consumption): 
        '''
            Sets parameter key_consumption. 
            Args: 
                key_consumption: string to define the key_consumption parameter 
        '''
        assert isinstance(key_consumption, str) or key_consumption is None, 'CycleHelper: Parameter key_conumption must be a string representation of the column name that encodes the kWh-consumption of data frames to be processed or None.'
        self.key_consumption = key_consumption
    
    def get_key_consumption(self): 
        '''
            Returns string of current key_consumption parameter. 
        '''
        return self.key_consumption

    def filter_out_micro_readings(self, df, threshold_divsion_factor=10.0, extend_basic_filtering=False, mode_multiplication_factor=1.2, fill_na=False):
        '''
            Since a heat pump never really switches off, but e.g. also the controlling units have an almost constant power consumption, there are a lot of readings that are very small, but not zero - even on separate HP smart meters.
            This method allows to filter out these readings before applying other analyses. It takes the maximum reading of the consumption into account to define a threshold. Below this threshold everything else will be cut off. 
            In case of extending the basic filtering, also the median and mode will be taken into account to potentially cut off even more - NOTE: ofen not recommended, because it might cut off too much - but give it a try. 

            Args: 
                df: data frame containing smart meter data 
                threshold_divsion_factor: a float defining a factor used for setting the cut off threshold - the higher the value the less aggressive is the filtering
                    NOTE: standard value 10.0 was chosen empirically and assumed kWh values, not kW
                extend_basic_filtering: Boolean to set to True if the basic filtering should be extended
                mode_multiplication_factor: a float defining a factor that is used to extend the basic filtering (in case it is activated)
                    NOTE: standard value 1.2 was again chosen empirically and assumed kWh values, not kW 
                fill_na: Boolean to set to True when the filtered out values should be filled with zero instead of NAN
            Returns: 
                a copy of the given smart meter data frame with filtered out max outliers (high readings occurring only one)
        '''
        assert isinstance(df, pd.DataFrame), 'CycleHelper.filter_out_micro_readings(): Parameter df must be of type pd.DataFrame.'
        assert self.key_consumption in df.columns.values, 'CycleHelper.filter_out_micro_readings(): Parameter df must contain a column named {}.'.format(self.key_consumption)
        assert isinstance(threshold_divsion_factor, float), 'CycleHelper.filter_out_micro_readings(): Parameter threshold_divsion_factor must be of type float.'
        assert isinstance(mode_multiplication_factor, float), 'CycleHelper.filter_out_micro_readings(): Parameter mode_multiplication_factor must be of type float.'
        assert isinstance(extend_basic_filtering, bool), 'CycleHelper.filter_out_micro_readings(): Parameter extend_basic_filtering must be of type boolean.'
        assert isinstance(fill_na, bool), 'CycleHelper.filter_out_micro_readings(): Parameter fill_na must be of type boolean.'

        df = df.copy()
        
        # based on the maximum consumption reading the threshold will be chosen
        max = df[self.key_consumption].max()

        if max > 1.0:
            threshold = 1.0 / threshold_divsion_factor
        elif max < 0.6:
            threshold = 0.6 / threshold_divsion_factor
        else: # any case there the maximum reading is between 0.6 and 1.0
            threshold = max /threshold_divsion_factor

        # cut off all values below threshold 
        # df[self.key_consumption] = df[self.key_consumption].where(df[self.key_consumption] > threshold)
        df.loc[df[self.key_consumption] <= threshold, self.key_consumption] = np.nan

        # in case of extending the basic filtering, the mode and median will be taken into account
        # if mode < median / 2: only keep values where the consumption is > mode * mode_multiplication_factor
        # NOTE: often not recommended, because it might cut away too much
        if extend_basic_filtering:
            mode = df[self.key_consumption].mode()[0]
            if mode < (df[self.key_consumption].median()/2):
                df[self.key_consumption] = df[self.key_consumption].where(df[self.key_consumption] > (mode*mode_multiplication_factor))
        if fill_na: 
            df[self.key_consumption] = df[self.key_consumption].fillna(0.0)
        return df

    def filter_out_max_outliers(self, df, fill_na=False):
        '''
            Here and there there are some very high readings that occur only once in a smart meter. These can be seen as some kind of glitches, sometimes causing problems to statistics, feature generation, etc.
            In order to filter these out before doing the analyses, this function can be used. 
            Args: 
                df: data frame containing smart meter data 
                fill_na: Boolean to set to True when the filtered out values should be filled with zero instead of NAN
            Returns: 
                a copy of the given smart meter data frame with filtered out max outliers (high readings occurring only one)
        '''
        assert isinstance(df, pd.DataFrame), 'CycleHelper.filter_out_max_outliers(): Cannot do filtering, because given parameter df is not of type pd.DataFrame, but of type: {}'.format(df)
        assert self.key_consumption in df.columns.values, 'CycleHelper.filter_out_max_outliers(): Parameter df must contain a column named {}.'.format(self.key_consumption)
        assert isinstance(fill_na, bool), 'CycleHelper.filter_out_micro_readings(): Parameter fill_na must be of type boolean.'

        df = df.copy()

        # replace max value with nan until the next max value occurs more than once -> get rid of max outliers
        while (df[df[self.key_consumption] == df[self.key_consumption].max()][self.key_consumption].count()) == 1:
            df[self.key_consumption] = df[self.key_consumption].where(df[self.key_consumption] < df[self.key_consumption].max())

        if fill_na: 
            df[self.key_consumption] = df[self.key_consumption].fillna(0.0)
        return df

    def filter_out_micro_readings_and_max_outliers(self, df, threshold_divsion_factor=10.0, extend_basic_filtering=False, mode_multiplication_factor=1.2, fill_na=False): 
        '''
            Performs filtering out micro reading and max outliers in combined way.
            For further explanations, please see documentation in functions: filter_out_max_outliers() and filter_out_micro_readings()
            NOTE: Especially recommended to be applied to separate smart meters. 
            Args: 
                df: data frame containing smart meter data 
                threshold_divsion_factor: a float defining a factor used for setting the cut off threshold 
                    NOTE: standard value 10.0 was chosen empirically and assumed kWh values, not kW
                extend_basic_filtering: Boolean to set to True if the basic filtering should be extended
                mode_multiplication_factor: a float defining a factor that is used to extend the basic filtering (in case it is activated)
                    NOTE: standard value 1.2 was again chosen empirically and assumed kWh values, not kW 
                fill_na: Boolean to set to True when the filtered out values should be filled with zero instead of NAN
            Returns: 
                a copy of the given smart meter data frame with filtered out max outliers and micro readings
        '''
        
        assert isinstance(df, pd.DataFrame), 'CycleHelper.filter_out_micro_readings_and_max_outliers(): Parameter df must be of type pd.DataFrame.'
        assert self.key_consumption in df.columns.values, 'CycleHelper.filter_out_micro_readings_and_max_outliers(): Parameter df must contain a column named {}.'.format(self.key_consumption)
        assert isinstance(threshold_divsion_factor, float), 'CycleHelper.filter_out_micro_readings_and_max_outliers(): Parameter threshold_divsion_factor must be of type float.'
        assert isinstance(mode_multiplication_factor, float), 'CycleHelper.filter_out_micro_readings_and_max_outliers(): Parameter mode_multiplication_factor must be of type float.'
        assert isinstance(extend_basic_filtering, bool), 'CycleHelper.filter_out_micro_readings_and_max_outliers(): Parameter extend_basic_filtering must be of type boolean.'
        assert isinstance(fill_na, bool), 'CycleHelper.filter_out_micro_readings_and_max_outliers(): Parameter fill_na must be of type boolean.'
        
        df = self.filter_out_max_outliers(df, fill_na=fill_na)
        df = self.filter_out_micro_readings(df, threshold_divsion_factor=threshold_divsion_factor, extend_basic_filtering=extend_basic_filtering, mode_multiplication_factor=mode_multiplication_factor, fill_na=fill_na)
        # df = self.filter_out_max_outliers(df, fill_na=fill_na)
        return df

    def get_resolution_time_delta(self, df): 
        '''
            Calculates the resolution of smart meter data in form of an object of type time_delta.
            Args: 
                df: data frame to be analyzed
            Returns: 
                object of type time_delta
        '''
        assert isinstance(df, pd.DataFrame) and self.key_timestamp in df.columns.values, 'CycleHelper.get_smd_time_delta_resolution(): Parameter df must be of type pd.DataFrame and must contain a column named {}.'.format(self.key_timestamp)
        df_copy = df.copy()
        df_copy.sort_values(by=[self.key_timestamp], inplace=True)
        df_copy.drop_duplicates(subset=[self.key_timestamp], inplace=True)
        return df_copy[self.key_timestamp].diff().min()

    def get_resolution_minutes(self, df): 
        '''
            Calculates the resolution in minutes of the smart meter data measurements. 
            NOTE: it assumes that the minimum time_delta between measurements is the resolution, i.e. ignores missing time stamps.
            NOTE: only works if the resolution is below one hour. 
            Args: 
                df: data frame to be analyzed
        '''
        assert isinstance(df, pd.DataFrame) and self.key_timestamp in df.columns.values, 'CycleHelper.get_smd_minutes_resolution(): Parameter df must be of type pd.DataFrame and must contain a column named {}.'.format(self.key_timestamp)
        time_delta = self.get_resolution_time_delta(df)
        minutes = time_delta.total_seconds()/60.0
        assert minutes <= 60.0, 'CycleHelper.get_smd_minutes_resolution() cannot get the resolution of the minutes because the time_delta between the measurements is larger than one hour.'
        return float(minutes)

    def get_kWh_to_kW_factor(self, df): 
        '''
            Calculates the factor to multiply the kWh consumptions with to turn them into kW values. 
            NOTE: only works if the resolution of smart meter data is below one hour.
            Args: 
                df: data frame to be analyzed
        '''
        assert isinstance(df, pd.DataFrame) and self.key_timestamp in df.columns.values, 'CycleHelper.get_kWh_to_kW_factor(): Parameter df must be of type pd.DataFrame and must contain a column named {}.'.format(self.key_timestamp)
        return 60.0 / self.get_resolution_minutes(df)

    def calculate_switched_on_column(self, df): 
        '''
            Returns copy of the data frame including a new column to define if a consumer is switched on (i.e. if there is a consumption greater than zero or not NAN).
            NOTE: new column switched_on is in format of a step function format (values either 1 or 0).
            Args: 
                df: data frame to be processed
        '''
        assert isinstance(df, pd.DataFrame) and self.key_consumption in df.columns.values, 'CycleHelper.calculate_switched_on_column(): Parameter df must be of type pd.DataFrame and must contain a column named {}.'.format(self.key_consumption)
        df = df.copy()
        df['temp'] = df[self.key_consumption].fillna(0.0)
        df['switched_on'] = df['temp'] > 0
        df['switched_on'] = df['switched_on'].astype('int')
        return df.drop(columns=['temp'])

    def calculate_cycle_columns(self, df:pd.DataFrame, enforce_calculation:bool=False):
        '''
            Returns a copy of the smart meter data frame with additional columns about the cyclic behavior: 
                - switched_on: binary values indicating if the device is switched on or not 
                - on_transient: 1 means switching on, 0 means no transient 
                - off_transient: 1 means switching off, 0 means no transient
                - fraction_of_next_reading: ratio of reading t0 to reading t+1 
                - fraction_of_previous_reading: ratio of reading t0 to reading t-1
                - oph_factor: factor for each timestamp indicating the percentage of resolution minutes the device was switched on
            NOTE: it is recommended to first disaggregate consumption within a desired window
            NOTE: it is not checked if the time series has missing timestamps, which could lead to wrong values
            Args: 
                df: the data frame containing smart meter data to be processed
                enforce_calculation: Boolean to indicate if a calculation is enforced
                    NOTE: if this parameter is False, it means that the columns are not newly calculated if they are all available 
        ''' 

        assert self.key_consumption in df.columns.values and self.key_timestamp in df.columns.values, 'CycleHelper.calculate_cycle_columns(): Parameter df must must contain following columns: {}.'.format([self.key_consumption, self.key_timestamp])
        
        if all(c in df.columns.values for c in ['switched_on', 'on_transient', 'off_transient', 'fraction_of_next_reading', 'fraction_of_previous_reading', 'oph_factor']) and not enforce_calculation:
            return df 
        
        df = df.copy()
        df.dropna(subset=[self.key_timestamp], inplace=True) # drop the observations where the keytimestamp is nan
        df.sort_values(by=self.key_timestamp, inplace=True)
        df.reset_index(inplace=True, drop=True)
        
        # create the switched_on column
        df = self.calculate_switched_on_column(df)

        # find the on-off-transients by diff-operation 
        # the first value will be NAN, since no diff can be calculated --> we will just ignore it 
        # diff_temp == -1.0 means switched from on to off 
        # diff_temp == +1.0 means switched from off to on
        # diff_temp == 0.0 means state not changed / running 
        df['diff_temp'] = df['switched_on'].diff()

        # now the switch-off markers are one timestamp too late (i.e. they mark the measurement after the last measurement where the device was still switched on)
        # therefore we need to move them by one position, such that diff == -1 is one timestamp earlier
        # therefore create a new column to indicate if a switching operation took place - one column each for switch on and off 
        indices = df[df['diff_temp'] == -1.0].index.values
        df['on_transient'] = 0.0
        df['off_transient'] = 0.0
        df.loc[[index-1 for index in indices], 'off_transient'] = 1.0
        df.loc[df['diff_temp'] == 1.0, 'on_transient'] = 1.0
        df.drop(columns=['diff_temp'], inplace=True)

        # treat first reading correctly 
        if df.iloc[0]['switched_on'] == 1: 
            df.loc[0, 'on_transient'] = 1.0
            
        # treat last reading correctly 
        if df.iloc[-1]['switched_on'] == 1: 
            df.loc[len(df)-1, 'off_transient'] = 1.0

        # calculate the fractions of the next reading
        df['temp'] = np.roll(df[self.key_consumption].values, -1)
        indices = df[df['temp'] != 0].index
        df['fraction_of_next_reading'] = np.nan
        df.loc[indices, 'fraction_of_next_reading'] = df.loc[indices, self.key_consumption] / df.loc[indices, 'temp']
        df.loc[len(df)-1, 'fraction_of_next_reading'] = np.nan

        # calculate the fractions of the previous reading
        df['temp'] = np.roll(df[self.key_consumption].values, 1)
        indices = df[df['temp'] != 0].index
        df['fraction_of_previous_reading'] = np.nan
        df.loc[indices, 'fraction_of_previous_reading'] = df.loc[indices, self.key_consumption] / df.loc[indices, 'temp']
        df.loc[0, 'fraction_of_previous_reading'] = np.nan
        df.drop(columns=['temp'], inplace=True)

        # now calculate the operating hour factor in a separate column by considering the switch on and off transients 
        # a factor of 1.0 means that the whole 15 minute interval should be counted 
        # a factor of 0.5 means half the 15 minute interval should be counted, etc.
        df['oph_factor'] = 0.0
        df.loc[(df['switched_on'] == 1) & (df['on_transient'] == 0.0) & (df['off_transient'] == 0.0), 'oph_factor'] = 1.0 # where no switching operation, but switched on - consider full time slot
        df.loc[(df['on_transient'] == 1.0) & (df['off_transient'] == 0.0), 'oph_factor'] = df.loc[(df['on_transient'] == 1.0) & (df['off_transient'] == 0.0), 'fraction_of_next_reading'] # where only switching on occurs, consider fraction of the next time slot
        df.loc[(df['off_transient'] == 1.0) & (df['on_transient'] == 0.0), 'oph_factor'] = df.loc[(df['off_transient'] == 1.0) & (df['on_transient'] == 0.0), 'fraction_of_previous_reading'] # where only switching off occurs, consider fraction of the previous time slot
        df.loc[(df['off_transient'] == 1.0) & (df['on_transient'] == 1.0), 'oph_factor'] = 0.5 # where both switching on and off occurs, consider it to be half cycle

       # for safety reasons, ceil the values - no oph_factors greater than 1.0 allowed
       # e.g. when HP switches on with with full load and then regulated down in the next reading the fractions calculated can exceed 1
        df.loc[df['oph_factor'] > 1.0, 'oph_factor'] = 1.0

        return df
    
    def extract_cycles(self, df:pd.DataFrame, exact_calculation:bool=True, round_decimals:bool=True): 
        '''
            Returns a data frame where for each cycle the following information is listed: 
                - cycle number 
                - cycle start 
                - cycle end 
                - cycle length in hours (either exact calculation or not)
                - energy consumption sum
                - energy consumption mean 
                - energy consumption modulation range (max - min) 
                    NOTE: for readings that are considered to be full readings (not fractional readings - depending on settings about exact calculations)
            NOTE: it is recommended to first disaggregate consumption within a desired window
            NOTE: it is not checked if the time series has missing timestamps, which could lead to wrong values
            Args: 
                df: pandas data frame containing smart meter data 
                exact_calculation: Boolean to set to True, if the fractional readings (switch o/off transients) should be used to derive exact operating times 
                    NOTE: otherwise a fractional reading (on-off-transient) counts as full measurement
                    NOTE: might slow down the calculations by a bit
                round_decimals: Boolean to set to True if the returned float values should be rounded to two decimals
                
        '''
        # group the information by cycle 
        df_smd = self.calculate_cycle_columns(df)
        readings_per_hour = self.get_kWh_to_kW_factor(df)

        if exact_calculation: 
            relevant_col = 'oph_factor'
        else: 
            relevant_col = 'switched_on'

        df_cycles = pd.DataFrame(columns=['cycle_number', 'cycle_start', 'cycle_end', 'cycle_duration', 'cycle_energy_sum'])
        
        # NOTE: the number of cycle starts and ends should always be the same - this is taken care of by calculate_cycle_columns()
        # however, to ensure it even when considering different time windows, we need to double check the first and last reading 

        # treat first reading correctly 
        if df_smd.iloc[0]['switched_on'] == 1: 
            df_smd.loc[0, 'on_transient'] = 1.0
            
        # treat last reading correctly 
        if df_smd.iloc[-1]['switched_on'] == 1: 
            df_smd.loc[len(df)-1, 'off_transient'] = 1.0
        
        df_cycles['cycle_start'] = df_smd[df_smd['on_transient'] == 1.0][self.key_timestamp].values
        df_cycles['cycle_end'] = df_smd[df_smd['off_transient'] == 1.0][self.key_timestamp].values
        df_cycles['cycle_number'] = df_cycles.index.values

        # if the smart meter data is time-zone aware, we need to convert the timestamps in df_cycles - otherwise no comparison is possible 
        tz = df_smd[self.key_timestamp].dt.tz
        if tz is not None:
            df_cycles['cycle_start'] = df_cycles['cycle_start'].dt.tz_localize('Europe/Zurich', ambiguous='NaT', nonexistent='NaT')
            df_cycles['cycle_start'] = df_cycles['cycle_start'].dt.tz_convert(tz)
            df_cycles['cycle_end'] = df_cycles['cycle_end'].dt.tz_localize('Europe/Zurich', ambiguous='NaT', nonexistent='NaT')
            df_cycles['cycle_end'] = df_cycles['cycle_end'].dt.tz_convert(tz)

        # loop over cycles and fill values 
        for idx, row in df_cycles.copy().iterrows(): 
            
            # get the smart meter data reffering to the cycle 
            df_sub = df_smd[(df_smd[self.key_timestamp] >= row['cycle_start']) & (df_smd[self.key_timestamp] <= row['cycle_end'])]
            
            # calculate the duration 
            val = float(df_sub[relevant_col].sum() / readings_per_hour)
            df_cycles.loc[idx, 'cycle_duration'] = round(val, 2) if round_decimals else val

            # calculate the energy consumption 
            val = df_sub[self.key_consumption].sum()
            df_cycles.loc[idx, 'cycle_energy_sum'] = round(val, 2) if round_decimals else val
        
        # drop observations with zero duration 
        df_cycles = df_cycles[df_cycles['cycle_duration'] > 0.0]
        df_cycles.reset_index(inplace=True, drop=True)

        return df_cycles
    
    def calculate_metrics_from_cycles(self, df_cycles, round_decimals:bool=True):
        '''
            Calculates metrics from the cycles dataframe. 
            NOTE: it returns the statistics for the whole time range covered in df_cycles. 
            NOTE: This means, e.g. if you want daily statistics, you should process the calculation of cycles and metrics for each day separately, i.e. in bunches.
            Args: 
                df_cycles: (dataframe) dataframe with cycles as returned by extract_cycles()
                round_decimals: Boolean to set to True if the returned float values should be rounded to two decimals
            Returns: 
                df_metrics: (dataframe) dataframe with metrics 
        '''

        df_metrics = pd.DataFrame()
        
        # operating hours
        val = pd.Series(df_cycles['cycle_duration'].sum())
        df_metrics['operating_hours'] = round(val, 2) if round_decimals else val

        # number of cycles
        df_metrics['cycles'] = len(df_cycles)

        # ratio: cycles by operating hours 
        df_metrics['ratio_cycles_operating_hours'] = df_metrics['cycles'] / df_metrics['operating_hours']
        if round_decimals: 
            df_metrics['ratio_cycles_operating_hours'] = df_metrics['ratio_cycles_operating_hours'].round(2)
            
        # average cycle duration
        val = df_cycles['cycle_duration'].mean()
        df_metrics['average_cycle_length_hours'] = round(val, 2) if round_decimals else val

        # sum of energy consumption
        val = df_cycles['cycle_energy_sum'].sum()
        df_metrics['energy_sum_kWh'] = round(val, 2) if round_decimals else val

        return df_metrics 
    
    def __calculate_daily_cycling_metrics_single_row__(self, row, df_filtered:pd.DataFrame, round_decimals:bool=True): 
        '''
            Helper method for parallelized calculation of daily cycling metrics. 
            Processes just a single row, which refers to a single date. 
            Args: 
                df_filtered: (dataframe) dataframe with filtered smart meter data
            Returns:
                row with daily cycling metrics
        '''

        # filter the SMD for the current date, i.e. the current row 
        df_sub = df_filtered[df_filtered[self.key_timestamp].dt.date == row['date']]

        # extract cycles
        df_cycles = self.extract_cycles(df_sub, exact_calculation=True, round_decimals=round_decimals)

        # extract metrics from the cycles 
        df_metrics = self.calculate_metrics_from_cycles(df_cycles, round_decimals=round_decimals)

        # add information to row 
        for column in df_metrics.columns:
            row[column] = df_metrics[column].values[0]

        return row

    def calculate_daily_cycling_metrics(self, df, round_decimals:bool=True, parallelized:bool=False, num_processes:int=None): 
        '''
            Calculates the daily cycling metrics from a dataframe containing only HP smart meter data. 
            Args: 
                df: (dataframe) dataframe with smart meter data - NOTE: should only contain HP energy consumption
                round_decimals: (bool) if True, the resulting metrics will be rounded to 2 decimals
                parallelized: (bool) if True, the calculation will be parallelized
                num_processes: number of concurrently running jobs, i.e. parameter for parallelization --> None means all available CPUs are used
            Returns: 
                df_metrics: (dataframe) dataframe with daily metrics
        '''

        # -------------------------------------------------------------------------
        # Remove Baseload - Basically readings that should be ignored will be set to NAN
        # -------------------------------------------------------------------------

        # estimate baseload and filter out readings below this estimate
        df_filtered = self.filter_out_micro_readings(df)

        # double check that filtering worked out - i.e. that more nan values than before - otherwise apply stronger filtering 
        if len(df_filtered[df_filtered[self.key_consumption].isna()]) <= len(df[df[self.key_consumption].isna()]): 
            df_filtered = self.filter_out_micro_readings(df, extend_basic_filtering=True)

        # -------------------------------------------------------------------------
        # Extract daily cycling metrics 
        # -------------------------------------------------------------------------

        # create a data frame that contains all unique dates of the smart meter data 
        df_daily = pd.DataFrame(df_filtered[self.key_timestamp].dt.date.unique(), columns=['date'])
        # df_daily = df_daily.iloc[:10]

        # get the sequence to iterate over 
        rows = [row for idx, row in df_daily.iterrows()]

        # -------------------------------------------------------------------------
        # Extract daily cycling metrics - parallelized version 
        # -------------------------------------------------------------------------

        if parallelized: 
            from multiprocessing import Pool, cpu_count
            from functools import partial 

            # define the number of CPUs to be used 
            if num_processes is None: 
                num_processes = min(len(rows), cpu_count())
            else: 
                assert isinstance(num_processes, int), 'CycleHelper.calculate_daily_cycling_metrics(): Parameter num_processes must be of type int or None.'

            # create an instance for parallel processing 
            with Pool(num_processes) as pool: 

                # get the returned rows as pandas series
                new_rows = pool.map(partial(self.__calculate_daily_cycling_metrics_single_row__, df_filtered=df_filtered, round_decimals=round_decimals), rows)

        # -------------------------------------------------------------------------
        # Extract daily cycling metrics - non-parallelized version 
        # -------------------------------------------------------------------------
        else: 
            new_rows = []
            for row in rows:
                new_rows.append(self.__calculate_daily_cycling_metrics_single_row__(row, df_filtered=df_filtered, round_decimals=round_decimals))

        # -------------------------------------------------------------------------
        # Reconstruct daily cycling metrics and create one data frame from it 
        # -------------------------------------------------------------------------

        # concatenate the rows again to one data frame 
        df_daily = pd.DataFrame.from_records(new_rows)

        # sort elements
        df_daily.sort_values(by=['date'], inplace=True)
        df_daily.reset_index(drop=True, inplace=True)

        return df_daily
    

    def calculate_daily_HP_utilization(self, daily_energy_values, hp_nominal_power:float, round_decimals:bool=True): 
        '''
            Calculates the utilization of an HP (in %) based on the daily energy consumption and the nominal power of the HP. 
            NOTE: this is not yet normalized by degree day or temperature!
            Args: 
                daily_energy_values: sequence or single value of daily energy consumption (kWh)
                hp_nominal_power: nominal power of the HP (kW)
            Returns: 
                daily_utilization: sequence or single value of daily utilization (%)
        '''

        daily_utilization = daily_energy_values / (hp_nominal_power * 24.0) * 100
        if round_decimals: 
            daily_utilization = np.round(daily_utilization, 2)
        return daily_utilization 

    def calculate_daily_energy_intensity(self, daily_energy_values, floor_area_qm:float, round_decimals:bool=True): 
        '''
            Calculates the energy intensity of an HP (kWh / qm) based on the daily energy consumption and the heated floor area. 
            NOTE: this is not yet normalized by degree day or temperature!
            Args: 
                daily_energy_values: sequence or single value of daily energy consumption (kWh)
                floor_area_qm: heated floor area (qm) 
            Returns: 
                daily_energy_intensity: sequence or single value of daily energy intensity (kWh / qm)
        '''
        daily_energy_intensity = daily_energy_values / floor_area_qm
        if round_decimals: 
            daily_energy_intensity = np.round(daily_energy_intensity, 2)
        return daily_energy_intensity 
    
    def calculate_linear_regression_parameters_from_daily_metrics(self, df:pd.DataFrame, temp_col:str, metrics:list, temp_min:int=0, temp_max:int=12, min_temps:int=6): 
        '''
            For given daily metrics, calculates temperature curves in defined temperature range and afterwards calculates linear regression parameters for each metric.
            Args: 
                df: (pd.DataFrame) data frame containing daily metrics of a single HP 
                temp_col: (string) column in data frame defining average daily temperature 
                metrics: (list) list of metrics for which linear regression parameters should be calculated (must be columns in df)
                temp_min: (int) minimum temperature to be considered for linear regression
                temp_max: (int) maximum temperature to be considered for linear regression
                min_temps: (int) minimum number of different temperature values to be observed for linear regression
            Returns:
                df_models: (pd.DataFrame) data frame containing linear regression parameters for each metric
                NOTE: returns None if no linear regression parameters could be calculated due to not satisfying the required number of temperatures observations
        '''

        assert temp_col in df.columns.values, 'CycleHelper.calculate_linear_regression_parameters_from_daily_metrics(): Given temperature column [{}] not found in data frame!'.format(temp_col)
        assert all(isinstance(metric, str) for metric in metrics), 'CycleHelper.calculate_linear_regression_parameters_from_daily_metrics(): Given metrics must be of type string!'
        for metric in metrics: 
            metric in df.columns.values, 'CycleHelper.calculate_linear_regression_parameters_from_daily_metrics(): Given temperature column [{}] not found in data frame!'.format(temp_col)

        # -------------------------------------------------------------------------
        # Calculate temperature curves  (median vectors)
        # -------------------------------------------------------------------------

        # create copy of data frame 
        df = df.copy()

        # remove any observations that are not in the desired temperature range 
        df = df[df[temp_col].between(temp_min, temp_max, inclusive='both')]

        # if not temperature observations are left, return None
        if len(df) == 0: 
            return None 

        # NOTE: the calculation of the temperature curves assumes that the temperature values are integers
        # therefore, round the temperature values to integers for the case that they are not already integers
        df[temp_col] = df[temp_col].round(0)

        # calculate the median for each metric and temperature 
        df_medians = df.groupby(by=[temp_col]).median()
        df_medians.reset_index(inplace=True)
        df_medians = df_medians.sort_values(by=[temp_col])

        # -------------------------------------------------------------------------
        # Calculate linear regressions 
        # -------------------------------------------------------------------------

        # create data frame for storing results 
        df_models = pd.DataFrame()

        # loop over metrics
        for metric in metrics:
                
            # drop NAN observations 
            df_sub = df_medians.copy().dropna(subset=[temp_col, metric])

            # ensure minimum number of non-NAN temperature observations
            if len(df_sub) < min_temps:
                continue 

            # prepare the values for linear regression  
            x = df_sub[temp_col].values.reshape(-1, 1)
            y = df_sub[metric].values.reshape(-1, 1)
                    
            # fit a linear regression model 
            model = LinearRegression().fit(x, y)

            # get information about model and store 
            df_temp_results = pd.DataFrame()
            df_temp_results['metric'] = pd.Series(metric)
            df_temp_results['R2'] = model.score(x, y)
            df_temp_results['intercept'] = model.intercept_
            df_temp_results['slope'] = model.coef_
            df_models = pd.concat([df_models, df_temp_results], axis=0)

        df_models.reset_index(inplace=True, drop=True)
        
        # if no models could be calculated due to not satisfying the required number of temperatures observations, return None
        if len(df_models) == 0:
            return None 
        return df_models
    
    def calculate_r2_statistics_from_regression_parameters(self, df:pd.DataFrame): 
        '''
            Calculates R^2 statistics for each metric from regression parameters.     
            Args: 
                df: (data frame) data frame with regression parameters
            Returns:
                df_r2: (data frame) data frame with R^2 statistics for each metric
        '''
        for col in ['R2', 'metric']: 
            assert col in df.columns, 'CycleHelper.calculate_r2_statistics_from_regression_parameters(): Given data Frame must contain a column named: {}'.format(col)

        # create data frame to store R2-statistics 
        df_r2 = pd.DataFrame()

        # get available metrics automatically and loop over them 
        metrics = df['metric'].unique()

        for metric in metrics: 
            
            # select the right data 
            df_sub = df[df['metric'] == metric]
            
            # calculate statistics 
            df_temp = pd.DataFrame()
            df_temp['metric'] = pd.Series(metric)
            df_temp['R2_observations_count'] = df_sub['R2'].count()
            df_temp['R2_median'] = np.round(df_sub['R2'].median(), 2)
            df_temp['R2_mean'] = np.round(df_sub['R2'].mean(), 2)
            df_temp['R2_std'] = np.round(df_sub['R2'].std(), 2)
            df_r2 = pd.concat([df_r2, df_temp], axis=0)
            
        df_r2.reset_index(inplace=True, drop=True)
        return df_r2

    def __calculate_statistics_for_reasoning_from_local_outlier_factors__(self, df:pd.DataFrame): 
        '''
            Derives the statistics from local outlier factors calculate to perform reasoning if an outlier is considered as such due to which parameter - slope or intercept.
            Args: 
                df: (pd.DataFrame) data frame containing the regression parameters of multiple HPs
            Returns:
                df: (pd.DataFrame) data frame containing the statistics per metric and type (inlier or outlier)
        '''
        for col in ['LOF_outlier', 'metric', 'slope', 'intercept']: 
            assert col in df.columns.values, 'CycleHelper.__calculate_statistics_for_reasoning_from_local_outlier_factors__(): The data frame must contain a column named {}!'.format(col)

        # get the existing metrics and loop over them
        metrics = df['metric'].unique()

        # create data frame for statistics 
        df_statistics = pd.DataFrame()

        # calculate statistics across slopes and intercepts for each metric and population 
        for metric in metrics: 
            
            # select the right data 
            df_sub = df[df['metric'] == metric]
            df_inliers = df_sub.loc[df_sub['LOF_outlier'] == False]
            df_outliers = df_sub.loc[df_sub['LOF_outlier'] == True]
            
            # create temporary data frame
            df_temp = pd.DataFrame()
            df_temp['metric'] = pd.Series(metric) 
            
            # create one row for inliers
            df_temp['type'] = 'inlier'
            df_temp['slope_median'] = df_inliers['slope'].median()
            df_temp['slope_mean'] = df_inliers['slope'].mean()
            df_temp['slope_std'] = df_inliers['slope'].std()
            df_temp['intercept_median'] = df_inliers['intercept'].median()
            df_temp['intercept_mean'] = df_inliers['intercept'].mean()
            df_temp['intercept_std'] = df_inliers['intercept'].std()
            
            # create one row for outliers and for all 
            df_temp.loc[len(df_temp.index)] = [metric, 'outlier', df_outliers['slope'].median(), df_outliers['slope'].mean(), df_outliers['slope'].std(), df_outliers['intercept'].median(), df_outliers['intercept'].mean(), df_outliers['intercept'].std()]
            df_temp.loc[len(df_temp.index)] = [metric, 'all', df_sub['slope'].median(), df_sub['slope'].mean(), df_sub['slope'].std(), df_sub['intercept'].median(), df_sub['intercept'].mean(), df_sub['intercept'].std()]
            
            # add to statistics 
            df_statistics = pd.concat([df_statistics, df_temp], axis=0)
        
        df_statistics.reset_index(drop=True, inplace=True)
        
        return df_statistics

    def calculate_local_outlier_factors_from_regression_parameters(self, df:pd.DataFrame, metrics:list, lof_neighbors:int=15, lof_contamination:float='auto', min_r2:float=0.4, suppress_prints:bool=False): 
        '''
            Applies Local Outlier Factor (LOF) to the regression parameters of the HPs to identify outliers.
            This means that it adds additional columns to each data point and metric, defining the local outlier factor and if it is considered an outlier or not.
            NOTE: LOF is applied to each metric individually and the given data frame is expected to contain the parameters of multiple HPs under evaluation (not only a single HP)!
            Args: 
                df: (pd.DataFrame) data frame containing the regression parameters of multiple HPs
                metrics: (list) list of metrics that should be considered for the outlier detection
                lof_neighbors: (int) number of neighbors that should be considered for the LOF calculation
                lof_contamination: (float) percentage of outliers that should be considered for the LOF - auto means it determines it automatically
                min_r2: (float) minimum R^2 value that should be considered for the LOF calculation - only households with R^2 value >= x are considered 
                suppress_prints: (bool) if True, no prints are shown
            Returns:
                df: (pd.DataFrame) data frame containing the evaluation per HP under evaluation 
        '''
        # ---------------------------
        # Preparations
        # ---------------------------

        # assertions 
        for col in ['metric', 'R2', 'intercept', 'slope']: 
            assert col in df.columns.values, 'CycleHelper.perform_outlier_detection_on_regression_parameters(): The data frame must contain a column named {}!'.format(col)
        for metric in metrics:
            assert isinstance(metric, str), 'CycleHelper.perform_outlier_detection_on_regression_parameters(): The metric must be a string!'
            assert metric in df['metric'].unique(), 'CycleHelper.perform_outlier_detection_on_regression_parameters(): The metric {} is not contained in the data frame!'.format(metric)

        # create additional columns on the data frame 
        df = df.copy()
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)
        df['LOF_applied'] = False
        df['LOF_prediction'] = np.nan
        df['LOF_negative_outlier_factor'] = np.nan # the higher the score, the more normal - inliers tend to have score close to 1
        df['LOF_radius'] = np.nan # radius for plotting according to outlier factor
        df['LOF_outlier'] = np.nan

        # ---------------------------
        # Perform Outlier Detection 
        # ---------------------------

        # perform outlier detection for each metric
        for metric in metrics: 

            # select the right data 
            df_sub = df[df['metric'] == metric]

            # only consider households with a good fit in the R^2 values, e.g. above 0.5
            df_sub = df_sub[df_sub['R2'] >= min_r2] 

            # only use non-nan values 
            df_sub = df_sub[~df_sub['intercept'].isna()]
            df_sub = df_sub[~df_sub['slope'].isna()]
            
            # get the values to perform LOF on
            data = df_sub[['intercept', 'slope']].values

            # get LOF predictions 
            model = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination)
            predictions = model.fit_predict(data)
            df.loc[df_sub.index, 'LOF_prediction'] = predictions
            outlier_factors = model.negative_outlier_factor_
            df.loc[df_sub.index, 'LOF_negative_outlier_factor'] = outlier_factors
            df.loc[df_sub.index, 'LOF_radius'] = (outlier_factors.max() - outlier_factors) / (outlier_factors.max() - outlier_factors.min())
            num_outliers = np.count_nonzero(predictions.flatten()== -1) # NOTE: outliers are -1 and inliers = 1
            df.loc[df_sub.index, 'LOF_applied'] = True

            if not suppress_prints:
                print('{} - Total Observations: {}'.format(metric, data.shape[0]))
                print('{} - LOF Outliers Detected: {} [{}%]'.format(metric, num_outliers, np.round(100.0*num_outliers/data.shape[0], 2)))
                print('-'*20)

        # post-processing 
        df['LOF_outlier'] = df['LOF_prediction'].map({-1.0 : True, 1.0 : False})

        # ---------------------------
        # Add Outlier Reasoning
        # ---------------------------

        # calculate the statistics for LOF reasoning 
        df_statistics = self.__calculate_statistics_for_reasoning_from_local_outlier_factors__(df)

        # extend df by columns for reasoning 
        df['LOF_outlier_slope_evaluation'] = np.nan
        df['LOF_outlier_intercept_evaluation'] = np.nan
        df['LOF_outlier_combined_evaluation'] = np.nan

        # for each metric perform resoning 
        for metric in metrics: 
            
            # select the right data 
            df_outliers = df[(df['metric'] == metric) & (df['LOF_outlier'])].copy()

            # get the means and std lines from the inliers 
            slope_mean = df_statistics[(df_statistics['metric'] == metric) & (df_statistics['type'] == 'inlier')]['slope_mean'].iloc[0]
            intercept_mean = df_statistics[(df_statistics['metric'] == metric) & (df_statistics['type'] == 'inlier')]['intercept_mean'].iloc[0]
            slope_std = df_statistics[(df_statistics['metric'] == metric) & (df_statistics['type'] == 'inlier')]['slope_std'].iloc[0]
            intercept_std = df_statistics[(df_statistics['metric'] == metric) & (df_statistics['type'] == 'inlier')]['intercept_std'].iloc[0]

            # derive slope evaluation 
            df.loc[df_outliers['slope'].between(slope_mean-slope_std, slope_mean+slope_std, inclusive='both').index, 'LOF_outlier_slope_evaluation'] = 'okay'
            df.loc[df_outliers[df_outliers['slope'] < slope_mean-slope_std].index, 'LOF_outlier_slope_evaluation'] = 'low'
            df.loc[df_outliers[df_outliers['slope'] > slope_mean+slope_std].index, 'LOF_outlier_slope_evaluation'] = 'high'
            
            # derive intercept evaluation 
            df.loc[df_outliers['intercept'].between(intercept_mean-intercept_std, intercept_mean+intercept_std, inclusive='both').index, 'LOF_outlier_intercept_evaluation'] = 'okay'
            df.loc[df_outliers[df_outliers['intercept'] < intercept_mean-intercept_std].index, 'LOF_outlier_intercept_evaluation'] = 'low'
            df.loc[df_outliers[df_outliers['intercept'] > intercept_mean+intercept_std].index, 'LOF_outlier_intercept_evaluation'] = 'high'

            # compile reasons into one label 
            df.loc[df_outliers.index, 'LOF_outlier_combined_evaluation'] = 'slope: ' + df.loc[df_outliers.index, 'LOF_outlier_slope_evaluation'] + ' | intercept: ' + df.loc[df_outliers.index, 'LOF_outlier_intercept_evaluation']

        return df 
    
    def derive_evaluation_from_local_outlier_factors(self, df:pd.DataFrame, metrics:list, id_col:str):
        '''
            The df with local outlier factors derived from calculate_local_outlier_factors_from_regression_parameters() contains one row per metric and HP.
            This function aggregates the results to one row per HP and adds a column that indicates whether the HP is atypical or not by any of the desired metrics under consideration.
            NOTE: a column in the data frame needs to contain the unique IDs of the HPs, as they are used for the aggregation.
            Args: 
                df: (pd.DataFrame) data frame with local outlier factors 
                metrics: (list) list of metrics that should be considered for the evaluation 
                id_col: (string) name of the column that contains the HP IDs
            Returns:
                df_grouped: (pd.DataFrame) data frame with one row per HP and a column that indicates whether the HP is atypical or not by any of the desired metrics under consideration.
        '''
        # assert that the data frame conains the right columns
        for col in ['metric', 'LOF_applied', 'LOF_outlier', 'LOF_outlier_combined_evaluation']: 
            assert col in df.columns, 'CycleHelper.derive_evaluation_from_local_outlier_factors(): Given data frame must contain a column named {}.'.format(col)
        
        # assert that the unique column is indeed unique --> no duplicates in metrics allowed 
        assert isinstance(id_col, str) and id_col in df.columns.values, 'CycleHelper.derive_evaluation_from_local_outlier_factors(): Given data frame must contain an ID column named {} as specified by id_col.'.format(id_col)
        df_grouped = df[[id_col, 'metric']].groupby(by=[id_col]).agg(list).reset_index()
        df_grouped['num_metrics'] = df_grouped['metric'].apply(lambda x: len(x))
        df_grouped['num_unique_metrics'] = df_grouped['metric'].apply(lambda x: len(np.unique(x)))
        assert len(df_grouped[df_grouped['num_metrics']!=df_grouped['num_unique_metrics']]) == 0, 'CycleHelper.derive_evaluation_from_local_outlier_factors(): Given data frame contains duplicate metrics for at least one HP according to the given id_col. Please make sure that each HP is only associated with one metric once.'

        # create copy with only the desired metrics to be considered 
        df = df.copy()

        # only consider metrics that should be evaluated 
        df = df[df['metric'].isin(metrics)]

        # perform flattening of the data frame 
        df_grouped = df[df['LOF_applied']][[id_col, 'metric']].groupby(by=[id_col]).agg(list).reset_index().rename(columns={'metric': 'metrics_evaluated'})
        df_grouped['metrics_outlier'] = df[df['LOF_applied']][[id_col, 'LOF_outlier']].groupby(by=[id_col]).agg(list).reset_index()['LOF_outlier']
        df_grouped['metrics_outlier_reasoning'] = df[df['LOF_applied']][[id_col, 'LOF_outlier_combined_evaluation']].groupby(by=[id_col]).agg(list).reset_index()['LOF_outlier_combined_evaluation']
        df_grouped['atypical_behavior'] = df_grouped['metrics_outlier'].apply(lambda x: True in x)
        df_grouped.insert(1, 'atypical_behavior', df_grouped.pop('atypical_behavior'))
        return df_grouped