import pandas as pd
import re
import subprocess
from io import StringIO

def raw_txt_to_df(raw_txt, first_col_name='P#'):
    # raw_txt = [x[:-1] for x in raw_txt] # remove \n from end of each line.
    raw_txt = [x.strip() for x in raw_txt]
    # raw_txt[:] = [re.sub(" +", "\t", x) for x in raw_txt]
    first_col_index = raw_txt[0].find(first_col_name)
    raw_txt[0] = raw_txt[0][first_col_index:]
    # breakpoint()
    raw_txt = [' '.join(x.split()) for x in raw_txt]
    raw_txt = [x.split(' ') for x in raw_txt]

    # raw_txt[:] = [x for x in raw_txt if x]
    # breakpoint()
    df = pd.DataFrame.from_records(raw_txt[1:])#, columns=raw_txt[0])
    breakpoint()



class ContamModel():
    def __init__(self, prj_file_loc):
        """ init method imports an existing .prj file
        Limitations:
         - Only steady state simulations
         - Only single storey CONTAM models
         - Only natural ventilation openings only.

        Args:
            prj_file_loc (str): .prj file location
        """
        self.prj_file_loc = prj_file_loc
        self.import_prj_file()
        self.get_flow_paths()


    def run_model(self):
        pass
    def run_simread(self):
        pass

    def import_prj_file(self):
        """imports the current .prj file at prj_file_loc

        Returns:
            list: [description]
        """
        with open(self.prj_file_loc, 'r') as f:
            lines = f.readlines()
            # for i, line in enumerate(lines):
            #     print(f'{i}:', line)
        self.prj_file = lines
    
    def update_prj_file(self):
        pass

    def get_flow_paths(self):#
        self.flow_paths = {}
        start = False
        for i, line in enumerate(self.prj_file):
            if re.search(".*flow paths:.*\n", line):
                self.flow_paths['raw_start'] = i+1
                start= True
            elif start and re.search("-999\n", line):
                self.flow_paths['raw_end'] = i

                break
        self.flow_paths['raw'] = self.prj_file[self.flow_paths['raw_start']:self.flow_paths['raw_end']] 
        self.flow_paths['df'] = raw_txt_to_df(self.flow_paths['raw'])
    
    
    def set_flow_paths(self, paths, values):
        pass

    def get_zones(self):
        for line in self.prj_file:
            if re.search(".*zones:.*\n", line):
                breakpoint()
    
    def set_zone_temperature(self, zone, value):
        pass

    def get_airflow_path_types(self):
        for line in self.prj_file:
            if re.search(".*flow elements.*\n", line):
                breakpoint()

    def get_environmental_conditions(self):
        pass

    def set_environmental_conditions(self, conditions,values):
        pass
    

    def import_flow_rates(self, file_loc):
        pass
    def extract_ventilation_matrix(self):
        pass

if __name__ == '__main__':
    prj = '/Users/Tom/Box/NCS Project/models/stochastic_model/contam_files/My_first_attemt.prj'
    x = ContamModel(prj_file_loc=prj)