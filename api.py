from fastapi import FastAPI # type: ignore
import requests

app = FastAPI()

# Ergast API URL for driver standings
ergast_api_url = "https://ergast.com/api/f1/current/driverStandings.json"

# FastAPI route to fetch and return the current driver standings
@app.get("/driver-standings/")
def get_driver_standings():
    response = requests.get(ergast_api_url)
    
    if response.status_code == 200:
        data = response.json()
        standings = data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        return standings
    else:
        return {"error": f"Failed to retrieve data: {response.status_code}"}
