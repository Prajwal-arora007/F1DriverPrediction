import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Cache data fetching functions to improve performance
@st.cache_data
def fetch_drivers(season):
    """Fetch all drivers for a given season."""
    url = f"https://ergast.com/api/f1/{season}/drivers.json"
    response = requests.get(url)
    if response.status_code == 200:
        drivers = response.json()['MRData']['DriverTable']['Drivers']
        driver_list = [f"{d['givenName']} {d['familyName']}" for d in drivers]
        driver_mapping = {f"{d['givenName']} {d['familyName']}": d['driverId'] for d in drivers}
        return driver_list, driver_mapping
    else:
        st.error(f"Failed to fetch drivers: {response.status_code}")
        return [], {}

@st.cache_data
def fetch_race_schedule(season):
    """Fetch all races for a given season."""
    url = f"https://ergast.com/api/f1/{season}.json"
    response = requests.get(url)
    if response.status_code == 200:
        races = response.json()['MRData']['RaceTable']['Races']
        return [
            {
                'round': race['round'],
                'race_name': race['raceName'],
                'circuit_name': race['Circuit']['circuitName'],
                'location': f"{race['Circuit']['Location']['locality']}, {race['Circuit']['Location']['country']}",
                'date': race['date']
            }
            for race in races
        ]
    else:
        st.error(f"Failed to fetch races: {response.status_code}")
        return []

@st.cache_data
def fetch_qualifying_results(season, round_number, driver_id):
    """Fetch qualifying position for the driver."""
    url = f"https://ergast.com/api/f1/{season}/{round_number}/drivers/{driver_id}/qualifying.json"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return int(response.json()['MRData']['RaceTable']['Races'][0]['QualifyingResults'][0]['position'])
        except (IndexError, KeyError):
            return None
    return None

@st.cache_data
def fetch_driver_race_result(season, round_number, driver_id):
    """Fetch the driver’s race result."""
    url = f"https://ergast.com/api/f1/{season}/{round_number}/drivers/{driver_id}/results.json"
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()['MRData']['RaceTable']['Races'][0]['Results'][0]
        position = int(result['position'])
        points = float(result['points'])
        constructor_name = result['Constructor']['name']
        return position, points, constructor_name
    else:
        st.error(f"Failed to fetch race result: {response.status_code}")
        return None, None, None

@st.cache_data
def fetch_constructor_points(season, constructor_name):
    """Fetch constructor points for the given season."""
    url = f"https://ergast.com/api/f1/{season}/constructorStandings.json"
    response = requests.get(url)
    if response.status_code == 200:
        standings = response.json()['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
        for constructor in standings:
            if constructor['Constructor']['name'] == constructor_name:
                # Handle missing points safely using .get()
                return float(constructor.get('points', 0.0))
    return 0.0

# App Title and Sidebar Inputs
st.title("F1 Race Outcome Prediction System")
st.sidebar.header("Input Parameters")

# Season and Driver Selection
season = st.sidebar.selectbox('Select Season', ['2023', '2022', '2021', '2020'])
driver_list, driver_mapping = fetch_drivers(season)
race_list = fetch_race_schedule(season)
driver = st.sidebar.selectbox('Select Driver', driver_list)
driver_id = driver_mapping[driver]

# Race Selection
race_selection = st.sidebar.selectbox('Select Race', [f"Round {r['round']}: {r['race_name']}" for r in race_list])
selected_race = next(r for r in race_list if f"Round {r['round']}: {r['race_name']}" == race_selection)
race_round = selected_race['round']

# Fetch Data for Display
qualifying_position = fetch_qualifying_results(season, race_round, driver_id)
finish_position, driver_points, constructor_name = fetch_driver_race_result(season, race_round, driver_id)
constructor_points = fetch_constructor_points(season, constructor_name)

# Display Actual Data
st.subheader("Actual Race and Qualifying Data")
st.write(f"**Qualifying Position:** {qualifying_position}")
st.write(f"**Finishing Position:** {finish_position}")
st.write(f"**Driver Points:** {driver_points}")
st.write(f"**Constructor:** {constructor_name}")
st.write(f"**Constructor Points:** {constructor_points}")

# Manual Inputs for Prediction
st.sidebar.subheader("Prediction Parameters")
weather = st.sidebar.selectbox('Weather Condition', ['Dry', 'Wet'])
lap_time = st.sidebar.number_input('Lap Time (seconds)', min_value=60.0, max_value=120.0, value=90.0)
pit_stops = st.sidebar.slider('Number of Pit Stops', min_value=0, max_value=5, value=2)

# Prepare Features and Train Model Dynamically
if qualifying_position is not None and finish_position is not None:
    X = np.array([[qualifying_position, constructor_points, lap_time, pit_stops]])
    y = np.array([finish_position])  # Use actual finish position as the target

    model_choice = st.sidebar.selectbox('Choose Model', ['Random Forest', 'XGBoost'])
    if model_choice == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = XGBRegressor(n_estimators=100, random_state=42)

    # Train the Model and Make Prediction
    model.fit(X, y)
    predicted_position = model.predict(X)[0]

    # Display Prediction
    st.subheader("Prediction")
    st.write(f"**Predicted Position for {driver}:** {int(predicted_position)}")

else:
    st.warning("Qualifying and race data are required for prediction.")
