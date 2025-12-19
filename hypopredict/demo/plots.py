import hypopredict.compressor as comp
import pandas as pd

def plot_glucose_83():

    ID = 8
    person = {"ID": ID}
    person["glucose"] = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[ID - 1])
    person["hg_events"] = comp.identify_hg_events(
        person["glucose"], min_duration=15, threshold=3.9
    )
    person['hg_events'] = person['hg_events'].loc['2014-10-03 11:25':'2014-10-04 09:05']

    comp.plot_hg_events_plotly(person)


def plot_glucose_64():

    ID = 6
    person = {"ID": ID}

    person["glucose"] = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[ID - 1])
    person["hg_events"] = comp.identify_hg_events(
        person["glucose"], min_duration=15, threshold=3.9
    )
    person['hg_events'] = person['hg_events'].loc['2014-10-04 11:03:26':'2014-10-04 22:57:07']

    comp.plot_hg_events_plotly(person)
