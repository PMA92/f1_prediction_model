import fastf1 as f1
import pandas as pd
def weather_summary(session, prefix):
    w = session.weather_data
    if w is None or w.empty:
        return {}                      # missing weather must not cost the race
    return {
        f'{prefix}_rain_any':         bool(w.Rainfall.any()),
        f'{prefix}_rain_changed':     bool(w.Rainfall.nunique() > 1),
        f'{prefix}_track_temp_mean':  round(w.TrackTemp.mean(), 2),
        f'{prefix}_track_temp_range': round(w.TrackTemp.max() - w.TrackTemp.min(), 2),
        f'{prefix}_air_temp_mean':    round(w.AirTemp.mean(), 2),
        f'{prefix}_wind_speed_mean':  round(w.WindSpeed.mean(), 2),
    }

def build_year_data(year, session):
    session = session[session['EventFormat'] != 'testing']
    frames = []
    weather_rows = []
    for event in session.itertuples():
        try: 
            race = f1.get_session(year, event.RoundNumber, 'R')
            qual = f1.get_session(year, event.RoundNumber, 'Q')
            race.load(laps=False, weather=True, telemetry=False, messages=False)
            qual.load(laps=False, weather=True, telemetry=False, messages=False)
            #load and process race and qual data
            r = (race.results[['Abbreviation', 'DriverNumber', 'ClassifiedPosition', 'Position', 'GridPosition', 'Status', 'Laps', 'TeamName']].copy())
            q = (qual.results[['Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']].copy())
            #get qual times
            q.rename(columns={'Position': 'QualifyingPosition'}, inplace=True)
            for c in ['Q1', 'Q2', 'Q3']:
                q[c] = q[c].dt.total_seconds()
            #get weather for race and qual 
            cur_w = {'Year': year, 'Round': int(event.RoundNumber), 'EventName': event.EventName}
            cur_w.update(weather_summary(race, 'race'))
            cur_w.update(weather_summary(qual, 'qual'))
            weather_rows.append(cur_w)
            #build dataframes 
            df = pd.merge(r, q, on='Abbreviation', how='left', validate='one_to_one')
            df.insert(0, 'Year', year)
            df.insert(0, 'Round', event.RoundNumber)
            df.insert(0, 'EventName', event.EventName)
            frames.append(df)
        except Exception as e:
            print(f"An error occurred while processing event {event.EventName} in year {year}: {e}")
            continue
        
    if not frames:
        print(f"no races collected for {year} -- nothing written")
        return

    data = pd.concat(frames, ignore_index=True)
    data.to_excel(f'data_{year}.xlsx', index=False)
    pd.DataFrame(weather_rows).to_excel(f'weather_{year}.xlsx', index=False)
    print(f"{len(data)} driver-races over {data.groupby(['Year','Round']).ngroups} races -> data_{year}.xlsx + weather_{year}.xlsx")
