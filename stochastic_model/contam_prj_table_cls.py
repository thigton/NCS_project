import re
import pandas as pd

class ContamPrjSnippets():
    """Data class handles the different parts of the contam .prj files
    Will store the raw string formats to update .prj file, dataframes for easy
    manipulation in python.
    """
    def __init__(self, search_string, prj_file):
        """init the data class.
        handle the different parts of contam files (.prj etc)

        Args:
            search_string (str):  indicator to find where the table starts
            prj_file (list<string>): Raw .prj file to search
        """
        self.search_string = search_string
        start = False
        for i, line in enumerate(prj_file):
            if re.search(self.search_string, line):
                self.table_header_idx = i+1
                start= True
            elif start and re.search("-999\n", line):
                self.end_idx = i

                break
        self.column_headers = prj_file[self.table_header_idx]
        self.raw_data = prj_file[self.table_header_idx+1:self.end_idx]




    def __raw_txt_to_df(self, raw_txt, first_col_name='P#'):
        """Needs sorting out still

        Args:
            raw_txt ([type]): [description]
            first_col_name (str, optional): [description]. Defaults to 'P#'.
        """


        def prep_column_names(raw_str, first_col_name, headers_to_remove=None):
            pass
        # raw_txt = [x[:-1] for x in raw_txt] # remove \n from end of each line.
        raw_txt = [x.strip() for x in raw_txt] # strip whitespace from the start and end of each line
        # raw_txt[:] = [re.sub(" +", "\t", x) for x in raw_txt]
        first_col_index = raw_txt[0].find(first_col_name) # get the index of the first column name
        raw_txt[0] = raw_txt[0].replace('<cfdData[4]>', '') # remove any text in column names in between <> as this messes up alignment.
        # raw_txt[0] = re.sub('<(cfdData[4])>', '', raw_txt[0]) # remove any text in column names in between <> as this messes up alignment.
        raw_txt[0] = raw_txt[0][first_col_index:] # get the column names

        raw_txt = [' '.join(x.split()) for x in raw_txt]
        raw_txt = [x.split(' ') for x in raw_txt]
        df = pd.DataFrame.from_records(raw_txt[1:])#, columns=raw_txt[0])
        breakpoint()
        return df

    def __unit_values_to_tuple(self, idx_range):
        pass