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
        self.get_flow_paths()
        # self.import_flow_rates()


    def run_contamX(self):
        output = subprocess.run([f"{self.contam_exe_dir}contamx3", f"{self.contam_dir}{self.project_name}.prj"],
                                stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE)
        if output.returncode ==0:
            print('ContamX run successfully!')
            for line in output.stdout.decode('utf-8').split('\n'):
                print(line)
        else:
            print('ContamX did not run successfully!')
            print(output.stderr)
            exit()


    def run_simread(self):
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
            print('simread run successfully!')
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
                # for i, line in enumerate(lines):
                #     print(f'{i}:', line)
            self.prj_file = lines
        except FileNotFoundError:
            print('.prj file is not found. Please check that the directory and file locations are correct')

    def update_prj_file(self):
        pass

    def get_flow_paths(self):
        self.flow_paths = ContamPrjSnippets(search_string=".*flow paths:.*\n",
                                            prj_file=self.prj_file)

    def set_flow_paths(self, paths, values):
        pass

    def get_zones(self):
        self.zones = ContamPrjSnippets(search_string=".*zones:.*\n",
                                            prj_file=self.prj_file)

    def set_zone_temperature(self, zone, value):
        pass

    def get_airflow_path_types(self):
        self.airflow_path_types = ContamPrjSnippets(search_string=".*flow elements.*\n",
                                                    prj_file=self.prj_file)

    def get_environmental_conditions(self):
        pass

    def set_environmental_conditions(self, conditions,values):
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
    prj_dir = '/home/tdh17/Documents/BOX/NCS Project/models/stochastic_model/contam_files/'
    name = 'school_corridor'
    x = ContamModel(contam_exe_dir=contam_exe_dir,
                    contam_dir=prj_dir,
                    project_name=name)