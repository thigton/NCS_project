import re
import pandas as pd
import os



class ContamPrjSnippets():
    """Data class handles the different parts of the contam .prj files
    Will store the raw string formats to update .prj file, dataframes for easy
    manipulation in python.
    """
    def __init__(self, search_string, prj_file, snippet_type, first_column_name=None):
        """init the data class.
        handle the different parts of contam files (.prj etc)

        Args:
            search_string (str):  indicator to find where the table starts
            prj_file (list<string>): Raw .prj file to search
        """
        self.search_string = search_string
        self.first_column_name = first_column_name
        self.snippet_type = snippet_type
        self.__find_raw_data(prj_file=prj_file)
        if self.snippet_type not in ['paths','zones', 'flow_elements', 'environment_conditions']:
            raise ValueError(f"{self.snippet_type} is not an option for snippet type, options are ['paths','zones', 'flow_elements', 'environment_conditions']")
        elif self.snippet_type in ['paths','zones', 'environment_conditions']:
            self.df = self.__get_table_df()
        elif self.snippet_type == 'flow_elements':
            self.df = self.__get_flow_element_df()
 
        

    def __repr__(self) -> str:
        return repr(self.df)
        
    def __find_raw_data(self, prj_file):
        found = False
        for i, line in enumerate(prj_file):
            if re.search(self.search_string, line):
                self.search_string_idx = i
                found = True
            elif found and re.search("-999\n", line):
                self.end_idx = i
                break
        if self.snippet_type in  ['paths', 'zones']:
            self.column_headers = prj_file[self.search_string_idx+1]
            self.raw_data = prj_file[self.search_string_idx+2:self.end_idx]
            self.number_of_columns = len(self.raw_data[0].split())
        elif self.snippet_type == 'flow_elements':
            self.raw_data = prj_file[self.search_string_idx+1:self.end_idx]
        elif self.snippet_type == 'environment_conditions':
            self.column_headers = prj_file[self.search_string_idx]
            self.raw_data = [prj_file[self.search_string_idx+1]]
            self.number_of_columns = len(self.raw_data[0].split())



    def __save_tmp_file(self):
        self.tmp_file_loc = f'{os.getcwd()}/tmp.txt'
        with open(self.tmp_file_loc, 'w') as f:
            f.writelines(self.raw_data)

    def __get_table_df(self):
        self.__save_tmp_file()

        return pd.read_csv(self.tmp_file_loc,
                        delim_whitespace=True,
                        comment='!',
                        index_col=False,
                        dtype=str,
                        names=self.__prep_df_column_names()
                        )

    def __prep_df_column_names(self):
        """returns some formatted headers for the dataframe
        Note: IMPORTANT this method means that some column names 
        are assigned to the wrong column. This only affects columns
        towards the end about units and cfd. Needs updating if columns are required!

        Returns:
            list<str>: list of column headers
        """
        column_headers = self.column_headers.split()
        idx = column_headers.index(self.first_column_name)
        # if there are less headers than columns then we add some dummy column names to the end so we 
        # don't lose any data
        if len(column_headers) < self.number_of_columns:
            return column_headers[idx:] + [f'dummy {str(x)}' for x in range(self.number_of_columns - len(column_headers)+1)]
        # if we have too many column headers we just lose some off the end (these columns aren't important/won't be changed)
        else:
            return column_headers[idx:self.number_of_columns+1]
        

    def __get_flow_element_df(self):
        data = [x.strip() for x in self.raw_data]
        df = pd.DataFrame()
        for i in range(0, len(data), 3):
            element_type = data[i].split()[2]
            try:
                col_names = ['id', 'type_id', 'element_type', 'name', 'description'] + self.flow_element_value_dict[element_type]
            except KeyError:
                breakpoint()
            series = pd.Series(data[i].split() + [data[i+1]] + data[i+2].split(),
                               index=col_names, dtype=str,
            )
            df = pd.concat([df, series], axis=1)
        return df.T


    def update_raw_data(self):
        if self.snippet_type == 'flow_elements':
            first_line = self.df[['id','type_id','element_type','name']]
            first_line = first_line.values.tolist()
            second_line = self.df['description']
            second_line = second_line.values.tolist()
            third_line = self.df.drop(['id','type_id','element_type','name', 'description'], axis=1)
            third_line = third_line.values.tolist()
            raw_data = []
            for first, second, third in zip(first_line, second_line, third_line):
                raw_data.extend([f" {' '.join([str(x) for x in first if str(x) != 'nan'])}\n",
                                 f" {second}\n",
                                 f" {' '.join([str(x) for x in third if str(x) != 'nan'])}\n",
                                 ])
            self.raw_data = raw_data
        else:
            # change the values which have become floats back into integers
            # values = self.df.select_dtypes(include=np.number).applymap('{:g}'.format)0
            values = self.df.values.tolist()

            # change to a list of strings, ASSUMPTION that the number of whitespaces between each value doesn't matter
            self.raw_data = [f"{'  '.join([str(inner) for inner in outer if str(inner) != 'nan'])}\n" for outer in values]


    @property
    def flow_element_value_dict(self):
        """For the flow elements values are given without meaning on .prj,
        for each flow type the meaning of each value is given in order

        Returns:
            dict: flow element parameters.
        """
        return {
            'dor_door': ['lam', 'turb','expt','dTmin','ht','wd','cd','u_T','u_H','u_W'],
            'dor_pl2': ['lam', 'turb','expt','dH', 'ht', 'wd','cd','u_H','u_W'],
            'plr_orfc': ['lam', 'turb','expt','area', 'dia', 'coef','Re','u_A','u_D'],
        }