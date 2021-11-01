import os
import pandas as pd
import subprocess
import numpy as np
import itertools
import pickle
try:
    from classes.contam_snippets import ContamPrjSnippets, ContamPrjSnippetsEnvironmentConditions, ContamPrjSnippetsFlowElements
    from classes.contam_vent_mat import ContamVentMatrixStorage
except ModuleNotFoundError:
    from contam_snippets import ContamPrjSnippets, ContamPrjSnippetsEnvironmentConditions, ContamPrjSnippetsFlowElements
    from contam_vent_mat import ContamVentMatrixStorage



class ContamModel():
    def __init__(self, contam_exe_dir, contam_dir, project_name, simread_file_name=None):
        """ init method imports an existing .prj file
        Limitations:
         - Only steady state simulations
         - Only single storey CONTAM models
         - Only natural ventilation openings only.

        Args:
            contam_exe_dir (str): contam executable file directory
            contam_dir (str): contam project file directory
            project_name (str): contam project name and file name
            simread_file_name (str): [optional] file name for the parameters required by simread.
        """
        self.contam_exe_dir = contam_exe_dir
        self.contam_dir = contam_dir
        self.project_name = project_name
        self.simread_file_name = simread_file_name

        self.__import_prj_file()
        self.__parse_flow_paths()
        self.__parse_zones()
        self.__parse_airflow_path_types()
        self.__parse_environment_conditions()

    def __repr__(self):
        return f'''Simulation details:- \n
Ambient temperature : {self.environment_conditions.df["Ta"].values[0]}K
Wind speed : {self.environment_conditions.df["Ws"].values[0]}m/s
Wind direction : {self.environment_conditions.df["Wd"].values[0]} deg.
          '''

    def run_simulation(self, verbose=False):
        """ Updates all changes to .prj file.
        Runs contam.
        Imports flow rates and builds ventilation matrix

        Args:
            verbose (bool, optional): Defaults to False.
        """
        if verbose:
            print(self)
        self.__update_prj_file()
        self.__run_contamX(verbose=verbose)
        self.__run_simread(verbose=verbose)
        self.__import_flow_rates()
        self.ventilation_matrix(verbose=verbose)

    def __run_contamX(self, verbose):
        """runs CONTAM

        Args:
            verbose (bool): print extra information to console.
        """
        output = subprocess.run([f"{self.contam_exe_dir}contamx3", f"{self.contam_dir}{self.project_name}.prj"],
                                stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE)
        if output.returncode != 0:
            print('ContamX did not run successfully!')
            print(output.stderr)
            exit()
        elif verbose:
            print('ContamX ran successfully!')
            for line in output.stdout.decode('utf-8').split('\n'):
                print(line)

    def __run_simread(self, verbose):
        """run simread to get the .lfr file output (flow rates in a useable format).
        """
        if not self.simread_file_name:
            print('No simread parameter file linked. Creating default file simread_parameters.txt')
            self.simread_file_name = 'simread_parameters'
            with open(f'{self.contam_dir}{self.simread_file_name}.txt', 'w') as f:
                f.writelines(['y\n','0-1000\n','y\n','1-1000\n'])
            # make a simread parameter file assuming you  and save it
        output = subprocess.run([f"'{self.contam_exe_dir}simread' '{self.contam_dir}{self.project_name}.sim' < '{self.contam_dir}{self.simread_file_name}.txt'"],
                                shell=True,
                                stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE)
        if output.returncode!=0:
            print('Simread did not run successfully!')
            print(output.stderr)
            exit()
        elif verbose:
            print('simread ran successfully!')
            for line in output.stdout.decode('utf-8').split('\n'):
                print(line)

    def __import_prj_file(self):
        """imports the current .prj file at prj_file_loc
        """
        try:
            with open(f'{self.contam_dir}{self.project_name}.prj', 'r') as f:
                lines = f.readlines()
            self.prj_file = lines
        except FileNotFoundError:
            print('.prj file is not found. Please check that the directory and file locations are correct')


    def __save_prj_file(self):
        """save the current version of the prj file
        """
        with open(f'{self.contam_dir}{self.project_name}.prj', 'w') as f:
            f.writelines(self.prj_file)

    def __update_prj_file(self):
        """update the raw prj file with the latest versions of the different dataframes
        """
        # flow paths
        self.flow_paths.update_raw_data()
        self.prj_file[self.flow_paths.search_string_idx+2:self.flow_paths.end_idx] = self.flow_paths.raw_data
        # zones
        self.zones.update_raw_data()
        self.prj_file[self.zones.search_string_idx+2:self.zones.end_idx] = self.zones.raw_data
        # airflow path types
        self.airflow_path_types.update_raw_data()
        self.prj_file[self.airflow_path_types.search_string_idx+1:self.airflow_path_types.end_idx] = self.airflow_path_types.raw_data
        # environment conditions
        self.environment_conditions.update_raw_data()
        self.prj_file[self.environment_conditions.search_string_idx+1] = self.environment_conditions.raw_data[0][:-1] + ' ! steady simulation \n'
        # save
        self.__save_prj_file()

    def __parse_flow_paths(self):
        """Extracts the flow path type from the .prj file
        """
        self.flow_paths = ContamPrjSnippets(search_string=".*flow paths:.*\n",
                                            first_column_name='P#',
                                            prj_file=self.prj_file)

    def set_initial_settings(self, weather, window_height):
        """Sets the initial conditions for the simulation.
        Use in all run scripts before run_simulation 
        to ensure the .prj file starts in the same point.

        Args:
            weather (Weather): initial Weather conditions
            window_height (float): initial window height.
        """
        # set weather params in model
        self.set_environment_conditions(condition='wind_direction', value=weather.wind_direction,  units='deg')
        self.set_environment_conditions(condition='wind_speed', value=weather.wind_speed,  units='km/hr')
        self.set_environment_conditions(condition='ambient_temp', value=weather.ambient_temp,  units='C')
        # ensure all windows are the same height and the right type
        self.set_all_flow_paths_of_type_to(search_type_term='indow',
                                                                param_dict={'opening_height': window_height,
                                                                            'type': 1}
                                                                            )
        # ensure all doorways are open at the start
        self.set_all_flow_paths_of_type_to(search_type_term='oor',
                                                                param_dict={'type': 4}
                                                                            )
        self.set_big_vent_matrix_idx(idx=-1) # All doors open
        corridor = self.get_all_zones_of_type(search_type_term='corridor')
        classrooms = self.get_all_zones_of_type(search_type_term='classroom')
        for zone in corridor['Z#'].values:
            self.set_zone_temperature(zone=zone, value=19.0)
        for zone in classrooms['Z#'].values:
            self.set_zone_temperature(zone=zone, value=21.0)


    def get_all_flow_paths_of_type(self, search_type_term,):
        """Get all flow paths of type

        Args:
            search_type_term (string): useful string to search and find all airflow types
                                    you want to change i.e. 'indow' -> 'Window' , 'oor' -> 'Door'.
                                    requires clear naming in the CONTAM file!

        Returns:
            pd.Dataframe: all flow paths of type (either door or window)
        """
        path_type_ids = self.airflow_path_types.df[self.airflow_path_types.df['name'].str.contains(search_type_term)]['id'].values
        return self.flow_paths.df[self.flow_paths.df['e#'].isin(path_type_ids)]


    def set_all_flow_paths_of_type_to(self, search_type_term, param_dict, rerun=False):
        """sets all the flow paths of a type

        Args:
            search_type_term (string): useful string to search and find all airflow types
                           you want to change i.e. 'indow' -> 'Window' , 'oor' -> 'Door'
            param_dict ({param: value}): [description]
        """
        path_ids = self.get_all_flow_paths_of_type(search_type_term)['P#'].values
        for path in path_ids:
            for p, v in param_dict.items():
                self.set_flow_path(path=int(path), param=p, value=v)
        if rerun:
            self.run_simulation(verbose=False)

    def set_flow_path(self, path, param, value):
        """can change the relative height of an opening and the type of the opening.
        The opening type must already be defined within the .prj file.
        NOTE: when changing the height there is not check to see whether this is physically reasonable.
        i.e. the top of the opening is above the defined room height.
        NOTE: This is a bit of a messy method; worth splitting if developing further.
        TODO: create your own error classes, some are used inappropriately here.

        Args:
            path (int): path number
            param (string): parameter to change options: ['type', 'opening_height']
            value (float or int): new value, data type depends on param 'type': int, 'opening_height': float

        """
        if not isinstance(path, int) or not str(path).isnumeric():
            raise TypeError('Path should be referenced by either an integer or a string of an integer')
        elif param not in ['type', 'opening_height']:
            raise ValueError('Only type or opening_height accepted in the param argument')
        elif param == 'opening_height' and not isinstance(value, float):
            raise TypeError('value should be a float if changing the opening height')
        elif param == 'type' and not isinstance(value, int):
            raise TypeError('value should be a int if changing the opening type')
        elif param == 'type' and str(value) not in self.airflow_path_types.df['id'].unique():
            raise ValueError(f'only opening types {self.airflow_path_types.df["id"].unique()} are available. Create new types directly in CONTAM and re-initialise.')
        elif len(self.flow_paths.df.loc[self.flow_paths.df['P#'] == str(path), 'P#']) == 0:
            raise KeyError('Path number not found. Either check for correct reference or create the path in CONTAM directly, and re-initialise.')

        # relating the param terms used to the df column names
        search_on = {'type': ['e#', str(value)],
                     'opening_height': ['relHt', f'{value:0.3f}'],
                     }
        self.flow_paths.df.loc[self.flow_paths.df['P#'] == str(path), search_on[param][0]] = search_on[param][1]

    def __parse_zones(self):
        """Extracts the zones table from the .prj file
        """
        self.zones = ContamPrjSnippets(search_string=".*zones:.*\n",
                                            first_column_name='Z#',
                                            prj_file=self.prj_file)

    def get_all_zones_of_type(self, search_type_term,):
        """Get all zones of type

        Args:
            search_type_term (string): useful string to search and find all airflow types
                                    you want to change i.e. 'Classroom' or 'Corridor'
                                    requires clear naming in the CONTAM file!

        Returns:
            pd.Dataframe: all zones of type.
        """
        return self.zones.df[self.zones.df['name'].str.contains(search_type_term[1:])]

    def get_zone_temp_of_room_type(self, search_type_term):
        """Get temperature of either the classrooms or the corridor.
        NOTE: Limitation: Forces all the rooms of the same type to be the same temperature.
        """
        zone_temperatures = self.get_all_zones_of_type(search_type_term)['T0']
        if zone_temperatures.nunique() != 1:
            raise ValueError(f'Not all {search_type_term}(s) have the same zone temperature. Error')
        else:
            return kelvinToCelcius(float(zone_temperatures.unique()[0]))

    def set_zone_temperature(self, zone, value, units='C'):
        """change the temperature of any zone in the dataframe
        TODO: For the units I have just used the default units set for my contam file.
        This must appear somewhere in the .prj file and should be checked to ensure
        changing to the right units.

        Args:
            zone (int, string): zone can be identified by either its number as an integer or a string. Or its name.
            value (float): new temperature
            units (str, optional): units of the new temperature. Defaults to 'C'. Options [Celcius [C] Fahrenheit [F] or Kelvin [K]]

        Raises:
            ValueError: Correct unit type
            TypeError: correct zone data type
        """
        if not isinstance(value, float):
            raise TypeError('the new temperature should be a float')
        if units not in ['C', 'F', 'K']:
            raise ValueError('Only Celcius [C] Fahrenheit [F] or Kelvin [K] accepted')
        elif units == 'C':
            value = celciusToKelvin(value)
        elif units == 'F':
            value = fahrenheitToKelvin(value)
        if isinstance(zone, int) or str(zone).isnumeric():
            search_on = 'Z#'  # search based on zone number
        elif isinstance(zone, str):
            search_on = 'name' # search based on zone name
        else:
            raise TypeError('zone data type can not be recognised.')
        if len(self.zones.df.loc[self.zones.df[search_on] == str(zone), 'T0']) == 0:
            raise KeyError('Zone name or number not found. Either check for correct reference or create the zone in CONTAM directly, and re-initialise.')

        self.zones.df.loc[self.zones.df[search_on] == str(zone), 'T0'] = f'{value:0.2f}'

    def __parse_airflow_path_types(self):
        """Extracts the airflow path types from the .prj file
        """
        self.airflow_path_types = ContamPrjSnippetsFlowElements(search_string=".*flow elements.*\n",
                                                    prj_file=self.prj_file)

    def __parse_environment_conditions(self):
        """Extracts the environmental conditions from the .prj file
        """
        self.environment_conditions = ContamPrjSnippetsEnvironmentConditions(search_string=".*!\s+Ta\s+Pb\s+Ws.*\n",
                                                    first_column_name='Ta',
                                                    prj_file=self.prj_file)


    def set_environment_conditions(self, condition, value, units='km/hr'):
        """change either the wind speed, direction or ambient temperature

        Args:
            condition (string): options [wind_speed, wind_direction,  ambient_temp]
            value (float): new value
            units (str, optional): units of value.
                                   wind options ['km/hr', 'm/s']
                                  temp options ['C', 'F', 'K']
                                  wind direction ['deg' , 'rad']
                                  Defaults to 'km/hr' and 'K'.
            NOTE: units are converted to the default value of the particular
            CONTAM file I am using, this must be referenced somewhere in the
            .prj file, so the correct units can be converted to regardless of the
            CONTAM file settings.

        Raises:
            TypeError: value must be a float
            ValueError: condition must be a valid option
            ValueError: units for wind must be a valid option
            ValueError: units for temp must be a valid option
        """
        if not isinstance(value, float):
            raise TypeError(f'the new {condition} should be a float')
        elif condition not in ['wind_speed', 'wind_direction', 'ambient_temp']:
            raise ValueError('Only wind_speed, wind_direction or ambient_temp can be changed.')
        elif units not in ['km/hr', 'm/s'] and condition == 'wind_speed':
            raise ValueError('Only kilometres/hr [km/hr] or metres/second [m/s] for wind speed')
        elif units not in ['rad','deg'] and condition == 'wind_direction':
            raise ValueError('Only degrees [deg] or redians [rad] for wind direction')
        elif units not in ['C', 'F', 'K'] and condition == 'ambient_temp':
            raise ValueError('Only Celcius [C] Fahrenheit [F] or Kelvin [K] accepted for ambient temp')
        elif units == 'km/hr':
            value = kilometresPerHourToMetresPerSecond(value)
        elif units == 'C':
            value = celciusToKelvin(value)
        elif units == 'F':
            value = fahrenheitToKelvin(value)
        elif units == 'rad':
            value = np.rad2deg(value)

        search_on = {'wind_speed': ['Ws', f'{value:0.3f}'],
                     'wind_direction': ['Wd', f'{value:0.1f}'],
                     'ambient_temp': ['Ta', f'{value:0.3f}']}
        self.environment_conditions.df.loc[0, search_on[condition][0]] = search_on[condition][1]

    def __import_flow_rates(self):
        """
        Understanding the volume flux direction.
         - The sign of the flux is from zone n# to m# on the flow paths table from the .prj file.
         - F0 will be populated first and F1 after if it is a bidirectional flow.
        """
        self.flow_rate_df = pd.read_csv(f'{self.contam_dir}{self.project_name}.lfr', sep='\t', header=0, index_col=2)

    def ventilation_matrix(self, verbose):
        """Produces a matrix of the venitlation rates as per equation 3.8
        Noakes and Sleigh (2009).
        NOTE: The data manipulation here is pretty messy combining the flow paths df and the
        flow rate df. A few of the matrices have been checked but more checking needed.
        NOTE: I am only using the 'single opening - two way flow model' on CONTAM. Need to check
        if there are any other bugs if different opening types are used.
        """
        no_zones = len(self.zones.df)
        self.vent_mat = np.empty(shape=(no_zones, no_zones))
        for i, j in itertools.product(range(no_zones), repeat=2):
            if i == j:
                # on the diagonal sum all flow rates out of the row zone (i)
                paths = self.search_flow_paths(zone_row_idx=i)
                into_zone = False
            else:
                paths = self.search_flow_paths(zone_row_idx=i, zone_col_idx=j)
                into_zone = True
            if isinstance(paths, float):
                self.vent_mat[i, j] = paths
            else:
                flow_rates = paths.apply(lambda x: self.choose_flow_rates(x, zone=self.zones.df.loc[i, 'Z#'], into_zone=into_zone), axis=1)
                self.vent_mat[i, j] = 0 - flow_rates.sum() if i == j else flow_rates.sum()
        np.savetxt(f"{self.contam_dir}vent_mat_check.csv", self.vent_mat, delimiter=",")
        self.vent_mat = kilogramPerSecondToMetresCubedPerHour(self.vent_mat)
        if verbose:
            print('Ventilation matrix produced...')

    def search_flow_paths(self, zone_row_idx, zone_col_idx=None):
        """used in self.ventilation_matrix()
        returns the flow paths associated with a zone and the flow rates

        Args:
            zone_row_idx (int): row in vent matrix (idx matches idx of self.zones.df)
            zone_col_idx (int, optional): column. Defaults to None (i.e. if on the diagonal)

        Returns:
            pd.Dataframe: dataframe of all flow paths incl. flow rates associated with a zone.
        """
        df = self.flow_paths.df
        zone_row = self.zones.df.loc[zone_row_idx, 'Z#']
        if zone_col_idx is not None:
            zone_col = self.zones.df.loc[zone_col_idx, 'Z#']
            paths = df.loc[((df['n#']==zone_row) & (df['m#']==zone_col)) | ((df['m#']==zone_row) & (df['n#']==zone_col)),:].copy()
        else:
            paths = df.loc[(df['n#']==zone_row) | (df['m#']==zone_row),:].copy()
        if len(paths) == 0:
            return 0.0
        else:
            paths['merge P#'] = paths['P#'].apply(lambda x: int(x))
            return paths.merge(right=self.flow_rate_df, right_index=True, left_on='merge P#')[['P#', 'n#','m#','F0 (kg/s)','F1 (kg/s)']]

    def choose_flow_rates(self, row, zone, into_zone=True):
        """ used in self.ventialtion_matrix() 
        given the flow paths and rates associated with the zone,
        get the correct flow rate / sign for the position in the vent matrix.
        Args:
            row (pd.Series): series of a flow path
            zone (int): zone id
            into_zone (bool, optional): Whether for the location on the ventilation matrix
            we are interested in the flow into or out of the zone. On diagonal, out of zone else into zone.
            Defaults to True.

        Returns:
            (pd.Series or pd.DataFrame): all ventilation rates which should be summed to form value in that position of the matrix
        """
        if (into_zone and row['m#'] == zone) or (not into_zone and row['n#'] == zone):
            # take the positive numbers
            return abs(row[['F0 (kg/s)', 'F1 (kg/s)']].max())
        else:
            # take the negative numbers
            return abs(row[['F0 (kg/s)', 'F1 (kg/s)']].min())

    def get_ventilation_rate_for(self, zone_name):
        """will return the total and fresh ventilation rates [litres per second]

        Args:
            zone_name (string): name of the zone

        Returns:
            tuple: total and fresh ventilation rates
        """
        zone = self.zones.df[self.zones.df['name'] == zone_name]
        zone_id = zone.index[0]
        flow_rates = self.search_flow_paths(zone_row_idx=zone_id)
        flow_rates['in'] = flow_rates.apply(lambda x: self.choose_flow_rates(x, zone=self.zones.df.loc[zone_id, 'Z#'], into_zone=True), axis=1)
        total = flow_rates['in'].sum()
        fresh = flow_rates[(flow_rates['n#'] == '-1') | (flow_rates['m#'] == '-1')]['in'].sum()
        return (kilogramPerSecondToLitresPerSecond(total) , kilogramPerSecondToLitresPerSecond(fresh))


    def generate_all_ventilation_matrices_for_all_door_open_close_combination(self, save=True):
        """will generate all a list of all possible ventilation matrices by opening and closing the doors.
        (12no. doors = 2**12 ventilation matrices).

        Args:
            save (bool, optional): to save or not to save. Defaults to True.
        """
        # get all flow paths which are internal doors
        all_doors = self.get_all_flow_paths_of_type(search_type_term='oor')
        external_doors = all_doors[(all_doors['n#'] =='-1') | (all_doors['m#'] =='-1')]
        self.external_door_matrix_idx = np.where(np.isin(all_doors['P#'].values, external_doors['P#'].values))[0]
        print(f'Number of matrices to be generated: {2**len(all_doors)}')
        self.all_door_matrices = []
        for i in range(2**len(all_doors)):
            # if i % 4 == 0:
                # print(f'Generating all matrices: {i/2**len(all_doors):0.1%}', end='\r')
            binary_ref = split(int_to_binary_ref(i,len(all_doors)))
            flow_path_types = [self.door_open if x == '1' else self.door_closed for x in binary_ref]
            for path, value in zip(all_doors['P#'].values, flow_path_types):
                self.set_flow_path(int(path), 'type', value)
            self.run_simulation(verbose=False)
            self.all_door_matrices.append(self.vent_mat)
        if save:
            self.save_all_ventilation_matrices()

    def save_all_ventilation_matrices(self):
        """Creates vent matrix storage object and saves it to a vent_mats folder in the contam_file directory.
        NOTE: could create a log of vent matrices produced
        """
        mat_obj = ContamVentMatrixStorage(self)
        fname = f'{mat_obj.date_init.strftime("%y%m%d_%H-%M")}_{self.project_name}.pickle'
        with open(f'{self.contam_dir}vent_mats/{fname}', 'wb') as pickle_out:
            pickle.dump(mat_obj, pickle_out)


    def get_window_height(self):
        """return the window height above the floor used in the model.
        NOTE: This is assumed to be the same for all windows

        Returns:
            float: window height
        """
        all_window_heights = self.get_all_flow_paths_of_type(search_type_term='indow')['relHt']
        if len(all_window_heights.unique()) != 1:
            print('Not all windows are the same height. At the moment this is a problem.')
            exit()
        else:
            return all_window_heights.unique()[0]
        
    def get_window_dimensions(self, which):
        """return the distance between the top and bottom of the window
        NOTE: This is assumed to be the same for all windows

        Returns:
            float: window height
        """
        dic = {'height': 'ht', 'width' : 'wd'}
        all_window_types = self.get_all_flow_paths_of_type(search_type_term='indow')['e#']
        def match_to_airflow_type_df(typ, typ_df, dim):
            return typ_df.loc[typ_df['id'] == typ, dim]
        extents = all_window_types.apply(lambda x: match_to_airflow_type_df(x, self.airflow_path_types.df, dic[which]))
        if len(extents[0].unique()) != 1:
            print(f'Not all windows have the same {which}. At the moment this is a problem.')
            exit()
        else:
            return float(extents[0].unique()[0])
    
    def set_big_vent_matrix_idx(self, idx):
        """Index of the current ventilation matrix in the simulation

        Args:
            idx (int): [description]
        """
        self.big_vent_matrix_idx = idx


    def load_all_vent_matrices(self,):
        """get all ventilation matrices for a particular set of environmental conditions.
        If they don't exist, create them and save.
        """
        window_height = self.get_window_height()
        Ta, _, Ws, Wd = self.environment_conditions.df.iloc[0,:4].values
        corridor_temp = self.get_zone_temp_of_room_type('corridor')
        classroom_temp = self.get_zone_temp_of_room_type('classroom')
        files = os.listdir(f'{self.contam_dir}/vent_mats/')   
        match = False
        for file in files:
            with open(f'{self.contam_dir}/vent_mats/{file}', 'rb') as pickle_in:
                obj = pickle.load(pickle_in)
            
            if (obj.contam_model_name == self.project_name and obj.window_height== window_height and 
                obj.ambient_temp == Ta and obj.wind_speed == Ws and obj.wind_direction == Wd and 
                obj.corridor_temp == corridor_temp and obj.classroom_temp == classroom_temp):
                self.all_door_matrices = obj.matrices
                self.external_door_matrix_idx = obj.external_door_matrix_idx
                match = True
        if not match:
            print(f'''File for project {self.project_name}, ambient temp: {Ta}, windspeed: {Ws},wind direction: {Wd},
                  corridor temp {corridor_temp}, classroom temp {classroom_temp}''')
            print('Not Found......Generating Now.')

            self.generate_all_ventilation_matrices_for_all_door_open_close_combination()

    
    @property
    def door_open(self):
        """The is the path type id in the .prj file for the door being open
        """
        return 4

    @property
    def door_closed(self):
        """The is the path type id in the .prj file for the door being closed
        """
        return 3


def int_to_binary_ref(i, string_width):
    """integer to a binary string.

    Args:
        i (int): index in list of ventilation matrices
        string_width (int or float): Length of the binary string (adding leading zeros)

    Returns:
        string: binary reference.
    """
    return "{0:b}".format(i).zfill(string_width)

def binary_ref_to_int(bin):
    """inverse function of int_to_binary_ref"""
    return int(bin, 2)

def split(string):
    return [char for char in string]

def kelvinToCelcius(T):
    return T - 273.15

def celciusToKelvin(T):
    return T + 273.15

def fahrenheitToKelvin(T):
    return (T - 32)*5/9 + 273.15

def kilometresPerHourToMetresPerSecond(v):
    return v / 3.6

def metresPerSecondToKilometresPerHour(v):
    return v*3.6

def kilogramPerSecondToMetresCubedPerHour(Q):
    air_density = 1.204
    return Q / air_density * 60**2

def kilogramPerSecondToLitresPerSecond(Q):
    air_density = 1.204
    return Q / air_density * 1e3

def PerHourToPerSecond(Q):
    return Q / 60**2

if __name__ == '__main__':
    contam_model_details = {'exe_dir': '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/',
                        'prj_dir': '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/',
                        'name': 'school_corridor'}

    contam_model = ContamModel(contam_exe_dir=contam_model_details['exe_dir'],
                contam_dir=contam_model_details['prj_dir'],
                project_name=contam_model_details['name'])

    contam_model.load_all_vent_matrices()
