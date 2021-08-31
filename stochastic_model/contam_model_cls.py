from os import POSIX_FADV_NOREUSE
import pandas as pd
import subprocess
import numpy as np
import itertools
from contam_prj_table_cls import ContamPrjSnippets







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
    def run_simulation(self):
        print(self)
        self.__update_prj_file()
        self.__run_contamX()
        self.__run_simread()
        self.__import_flow_rates()
        self.ventilation_matrix()
        

    def __run_contamX(self):
        output = subprocess.run([f"{self.contam_exe_dir}contamx3", f"{self.contam_dir}{self.project_name}.prj"],
                                stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE)
        if output.returncode ==0:
            print('ContamX ran successfully!')
            for line in output.stdout.decode('utf-8').split('\n'):
                print(line)
        else:
            print('ContamX did not run successfully!')
            print(output.stderr)
            exit()


    def __run_simread(self):
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
        if output.returncode==0:
            print('simread ran successfully!')
            for line in output.stdout.decode('utf-8').split('\n'):
                print(line)
        else:
            print('Simread did not run successfully!')
            print(output.stderr)
            exit()

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
                                            snippet_type='paths',
                                            prj_file=self.prj_file)


    def set_flow_paths(self, path, param, value):
        """can change the relative height of an opening and the type of the opening.
        NOTE: when changing the height there is not check to see whether this is physically reasonable.
        i.e. the top of the opening is above the defined room height.
        NOTE: This is a bit of a messy method; worth splitting if developing further.

        Args:
            path (int or string of int): path number
            param (string): parameter to change options: ['type', 'opening_height']
            value (float or int): new value, data type depends on param 'type': int, 'opening_height': float

        Raises:
            TypeError: [description]
            ValueError: [description]
            TypeError: [description]
            TypeError: [description]
            ValueError: [description]
            KeyError: [description]
        """
        if not isinstance(path, int) or not str(path).isnumeric():
            raise TypeError('Path should be referenced by either an integer or a string of an integer')
        elif param not in ['type', 'opening_height']:
            raise ValueError('Only param type or opening_height accepted')
        elif param == 'opening_height' and not isinstance(value, float):
            raise TypeError('value should be a float if changing the opening height')
        elif param == 'type' and not isinstance(value, int):
            raise TypeError('value should be a int if changing the opening type')
        elif param == 'type' and str(value) not in self.airflow_path_types.df['id'].unique():
            raise ValueError(f'only opening types {self.airflow_path_types.df["id"].unique()} are available. Create new types directly in CONTAM and re-initialise.')
        elif len(self.flow_paths.df.loc[self.flow_paths.df['P#'] == str(path), 'P#']) == 0:
            raise KeyError('Path number not found. Either check for correct reference or create the path in CONTAM directly, and re-initialise.')
        
        search_on = {'type': ['e#', str(value)],
                     'opening_height': ['relHt', f'{value:0.3f}'],
                     }
        self.flow_paths.df.loc[self.flow_paths.df['P#'] == str(path), search_on[param][0]] = search_on[param][1]

    def __parse_zones(self):
        """Extracts the zones table from the .prj file
        """
        self.zones = ContamPrjSnippets(search_string=".*zones:.*\n",
                                            first_column_name='Z#',
                                            snippet_type='zones',
                                            prj_file=self.prj_file)

    def set_zone_temperature(self, zone, value, units='C'):
        """change the temperature of any zone in the dataframe

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
        self.airflow_path_types = ContamPrjSnippets(search_string=".*flow elements.*\n",
                                                    snippet_type='flow_elements',
                                                    prj_file=self.prj_file)



    def __parse_environment_conditions(self):
        """Extracts the environmental conditions from the .prj file
        """
        self.environment_conditions = ContamPrjSnippets(search_string=".*!\s+Ta\s+Pb\s+Ws.*\n",
                                                    first_column_name='Ta',
                                                    snippet_type='environment_conditions',
                                                    prj_file=self.prj_file)

    def set_environment_conditions(self, condition, value, units='km/hr'):
        """change either the wind speed, direction or ambient temperature

        Args:
            condition (string): options [wind_speed, wind_direction,  ambient_temp]
            value (float): new value
            units (str, optional): units of value.
                                   wind options ['km/hr', 'm/s']
                                  temp options ['C', 'F', 'K']
                                  Defaults to 'km/hr'.

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
        elif units not in ['km/hr', 'm/s'] and condition in ['wind_speed', 'wind_direction']:
            raise ValueError('Only kilometres/hr [km/hr] or metres/second [m/s] for wind speed and direction')
        elif units not in ['C', 'F', 'K'] and condition == 'ambient_temp':
            raise ValueError('Only Celcius [C] Fahrenheit [F] or Kelvin [K] accepted for ambient temp')
        elif units == 'km/hr':
            value = kilometresPerHourToMetresPerSecond(value)
        elif units == 'C':
            value = celciusToKelvin(value)
        elif units == 'F':
            value = fahrenheitToKelvin(value)

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

    def ventilation_matrix(self):
        """Produces a matrix of the venitlation rates as per equation 3.8
        Noakes and Sleigh (2009). 
        NOTE: The data manipulation here is pretty messy combining the flow paths df and the
        flow rate df. A few of the matrices have been checked but more checking needed.
        NOTE: This function relies on the zone numbers being sequential starting at 1
        """
        def choose_flow_rates(row, zone, into_zone=True):
            if (into_zone and row['m#'] == zone) or (not into_zone and row['n#'] == zone):
                # take the positive numbers
                return abs(row[['F0 (kg/s)', 'F1 (kg/s)']].max())
            else:
                # take the negative numnbers
                return abs(row[['F0 (kg/s)', 'F1 (kg/s)']].min())

        no_zones = len(self.zones.df)
        self.vent_mat = np.empty(shape=(no_zones, no_zones))
        for i, j in itertools.product(range(no_zones), repeat=2):
            if i == j:
                # on the diagonal sum all flow rates out of the row zone (i)
                paths = self.search_flow_paths(zone_row=str(i+1))
                into_zone = False
            else:
                paths = self.search_flow_paths(zone_row=str(i+1), zone_col=str(j+1))
                into_zone = True
            if isinstance(paths, float):
                self.vent_mat[i, j] = paths
                continue
            flow_rates = paths.apply(lambda x: choose_flow_rates(x, zone=str(i+1), into_zone=into_zone), axis=1)
            self.vent_mat[i, j] = 0 - flow_rates.sum() if i == j else flow_rates.sum()
        np.savetxt(f"{self.contam_dir}vent_mat_check.csv", self.vent_mat, delimiter=",")
        self.vent_mat = kilogramPerSecondToMetresCubedPerHour(self.vent_mat)
        print('Ventilation matrix produced...')
    
    def search_flow_paths(self, zone_row, zone_col=None):
        if zone_col:
            paths = self.flow_paths.df.loc[((self.flow_paths.df['n#']==zone_row) & (self.flow_paths.df['m#']==zone_col)) | ((self.flow_paths.df['m#']==zone_row) & (self.flow_paths.df['n#']==zone_col)),:].copy()
        else:
            paths = self.flow_paths.df.loc[(self.flow_paths.df['n#']==zone_row) | (self.flow_paths.df['m#']==zone_row),:].copy()
        if len(paths) == 0:
            return 0.0
        else:
            paths['merge P#'] = paths['P#'].apply(lambda x: int(x))
            return paths.merge(right=self.flow_rate_df, right_index=True, left_on='merge P#')[['P#', 'n#','m#','F0 (kg/s)','F1 (kg/s)']]
    

def celciusToKelvin(T):
    return T + 273.15

def fahrenheitToKelvin(T):
    return (T - 32)*5/9 + 273.15

def kilometresPerHourToMetresPerSecond(v):
    return v / 3.6

def kilogramPerSecondToMetresCubedPerHour(Q):
    air_density = 1.204
    return Q / air_density * 60**2


if __name__ == '__main__':
    pass
