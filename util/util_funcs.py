import pandas as pd
import os
from datetime import datetime, timedelta

def load_model_log(fname):
    dtype_converters = {
        'duration': lambda x: timedelta(days=datetime.strptime(x,"%d days").day),
        'school_start': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'school_end': lambda x: datetime.strptime(x, '%H:%M:%S').time(),
        'time_step': lambda x: timedelta(minutes=datetime.strptime(x,"0 days %H:%M:%S").minute),
    }
    try:
        return pd.read_csv(fname,index_col=0, converters=dtype_converters)
    except FileNotFoundError:
        print(f'{fname} not found. Initialising a new model log')
        return pd.DataFrame()
    

def model_params_to_log_format(model,  fname):
    series = pd.Series(model.consts, name=fname)
    # wind_speed, wind_dir, amb_temp, corridor_temp, classroom_temp, window_height, opening_method, sim_method
    for attr in ['wind_speed', 'wind_direction', 'ambient_temp']:
        series[attr] = getattr(model.weather, attr)
    for room in ['corridor', 'classroom']:
        series[f'{room}_temp'] = model.contam_model.get_zone_temp_of_room_type(room)
    series['window_height'] = model.contam_model.get_window_height()
    series['opening_method'] = model.opening_method
    series['movement_method'] = model.movement_method
    series['method'] = model.method
    series['contam_model_name'] = model.contam_model.project_name
    return series


def add_model_to_log(model, df, fname):
    series = model_params_to_log_format(model, fname)
    return df.append(series.T)