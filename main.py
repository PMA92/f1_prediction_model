import fastf1 as f1

def get_races(year: int):
    race1 = f1.get_event(year, "Canadian Grand Prix")
    print(race1["Session5"])

get_races(2023)
