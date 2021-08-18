import pandas as pd
import re
import subprocess
from io import StringIO

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

        self.import_prj_file()
        self.parse_flow_paths()
        self.parse_zones()
        self.parse_airflow_path_types()
        self.parse_environment_conditions()


    def run_simulation(self):
        self.__update_prj_file()
        self.__run_contamX()
        self.__run_simread()

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

    def import_prj_file(self):
        """imports the current .prj file at prj_file_loc

        Returns:
            list: [description]
        """
        try:
            with open(f'{self.contam_dir}{self.project_name}.prj', 'r') as f:
                lines = f.readlines()
            self.prj_file = lines
        except FileNotFoundError:
            print('.prj file is not found. Please check that the directory and file locations are correct')

    def __save_prj_file(self):
        with open(f'{self.contam_dir}{self.project_name}.prj', 'w') as f:
            f.writelines(self.prj_file)

    def __update_prj_file(self):
        # update flow paths, zones, airflow_path_types and environmental conditions
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

        self.prj_file[self.environment_conditions.search_string_idx+1] = self.environment_conditions.raw_data
        # save
        self.__save_prj_file()
    def parse_flow_paths(self):
        self.flow_paths = ContamPrjSnippets(search_string=".*flow paths:.*\n",
                                            first_column_name='P#',
                                            snippet_type='table',
                                            prj_file=self.prj_file)
    def set_flow_paths(self, paths, values):
        pass

    def parse_zones(self):
        self.zones = ContamPrjSnippets(search_string=".*zones:.*\n",
                                            first_column_name='Z#',
                                            snippet_type='table',
                                            prj_file=self.prj_file)

    def set_zone_temperature(self, zone, value):
        pass

    def parse_airflow_path_types(self):
        self.airflow_path_types = ContamPrjSnippets(search_string=".*flow elements.*\n",
                                                    snippet_type='flow_elements',
                                                    prj_file=self.prj_file)



    def parse_environment_conditions(self):
        self.environment_conditions = ContamPrjSnippets(search_string=".*!\s+Ta\s+Pb\s+Ws.*\n",
                                                    first_column_name='Ta',
                                                    snippet_type='environment_conditions',
                                                    prj_file=self.prj_file)

    def set_environment_conditions(self, conditions,values):
        pass


    def import_flow_rates(self):
        """
        Understanding the volume flux direction.
         - The sign of the flux is from zone n# to m# on the flow paths table from the .prj file.
         - F0 will be populated first and F1 after if it is a bidirectional flow.
        """
        self.flow_rate_df = pd.read_csv(f'{self.contam_dir}{self.project_name}.lfr', sep='\t', header=0, index_col=2)

    def extract_ventilation_matrix(self):
        pass





if __name__ == '__main__':
    contam_exe_dir = '/home/tdh17/contam-x-3.4.0.0-Linux-64bit/'
    prj_dir = '/Users/Tom/Box/NCS Project/models/stochastic_model/contam_files/'
    name = 'school_corridor'
    x = ContamModel(contam_exe_dir=contam_exe_dir,
                    contam_dir=prj_dir,
                    project_name=name)
    x.run_simulation()