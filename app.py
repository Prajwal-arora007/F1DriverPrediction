import requests
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Cache API calls for performance
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
                'date': race['date']
            }
            for race in races
        ]
    else:
        return []

@st.cache_data
def fetch_driver_standings(season, round_number):
    """Fetch driver standings for a given season and round."""
    url = f"http://ergast.com/api/f1/{season}/{round_number}/driverStandings.json"
    response = requests.get(url)
    if response.status_code == 200:
        standings = response.json()['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
        return pd.DataFrame([{
            "Driver": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "Position": int(s['position']),
            "Points": float(s['points'])
        } for s in standings])
    else:
        return pd.DataFrame()

@st.cache_data
def fetch_lap_times(year, round_number):
    """Fetch lap times for a given race year and round."""
    url = f"http://ergast.com/api/f1/{year}/{round_number}/laps.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        try:
            lap_times_data = response.json()
            lap_times = []
            for race in lap_times_data['MRData']['RaceTable']['Races']:
                for lap in race['Laps']:
                    for timing in lap['Timings']:
                        driver_id = timing['driverId']
                        lap_time_str = timing.get('time', None)
                        
                        if lap_time_str:  # Ensure lap time exists
                            min_sec = lap_time_str.split(":")
                            if len(min_sec) == 2:
                                minutes, seconds = min_sec
                                total_seconds = int(minutes) * 60 + float(seconds)
                                lap_times.append({
                                    "Driver": driver_id,
                                    "Lap Time (s)": total_seconds,
                                    "Lap": lap['number']
                                })

            return pd.DataFrame(lap_times)
        
        except Exception as e:
            print(f"Error parsing lap times: {e}")
            return pd.DataFrame()

    else:
        print(f"Failed to fetch lap times, status code: {response.status_code}")
        return pd.DataFrame()

# Streamlit App
st.title("F1 Race Prediction System")
st.sidebar.header("Set Parameters")

# Season and Race Data
season = st.sidebar.selectbox("Select Season", ['2023', '2022', '2021', '2020'])
driver_list, driver_mapping = fetch_drivers(season)
race_schedule = fetch_race_schedule(season)

# Driver and Race Selection
driver = st.sidebar.selectbox("Select Driver", driver_list)
driver_id = driver_mapping[driver]
race_selection = st.sidebar.selectbox(
    "Select Race", [f"Round {r['round']}: {r['race_name']}" for r in race_schedule]
)
selected_race = next(r for r in race_schedule if f"Round {r['round']}: {r['race_name']}" == race_selection)
race_round = selected_race['round']

# Fetch Driver Standings Data
standings_df = fetch_driver_standings(season, race_round)

# Fetch Lap Times Data
lap_times_df = fetch_lap_times(season, race_round)

# Reset Prediction Parameters on Selection Change
if "last_driver" not in st.session_state or driver != st.session_state.last_driver:
    st.session_state.avg_lap_time = 90.0  # Set the initial average lap time
    st.session_state.pit_stops = 2
    st.session_state.track_condition = "Dry"
st.session_state.last_driver = driver

# Prediction Parameters
st.sidebar.subheader("Prediction Inputs")
avg_lap_time = st.sidebar.number_input(
    "Average Lap Time (seconds)", 
    min_value=60.0, max_value=120.0, value=st.session_state.avg_lap_time, step=0.1
)
pit_stops = st.sidebar.slider(
    "Number of Pit Stops", 
    min_value=0, max_value=5, value=st.session_state.pit_stops
)
track_condition = st.sidebar.selectbox(
    "Track Condition", 
    ['Dry', 'Wet'], 
    index=0 if st.session_state.track_condition == "Dry" else 1
)

# Original Race Data (Static)
original_data = {
    "Qualifying Position": standings_df[standings_df["Driver"] == driver]["Position"].values[0] 
    if not standings_df.empty else None,
    "Points": standings_df[standings_df["Driver"] == driver]["Points"].values[0] 
    if not standings_df.empty else None,
    "Lap Time (s)": st.session_state.avg_lap_time,  # Keep the original avg lap time
    "Pit Stops": 2  # Default
}

# Display Original Data Table (Original Lap Time is preserved)
st.subheader("Original Race Data")
st.table(pd.DataFrame([original_data]))

# Predictions based on parameters
if not standings_df.empty:
    # Simulate data for prediction
    dummy_X = np.random.rand(100, 3)  # 100 samples with 3 features
    dummy_y = np.random.randint(1, 21, 100)  # Random positions (1-20)

    # Train a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(dummy_X, dummy_y)

    # Predict position for the selected driver
    X_test = np.array([[original_data["Qualifying Position"], avg_lap_time, pit_stops]])
    predicted_position = int(model.predict(X_test)[0])

    # Predicted Race Data
    predicted_data = {
        "Lap Time (s)": avg_lap_time,
        "Pit Stops": pit_stops,
        "Predicted Position": predicted_position
    }

    # Display Predicted Data Table
    st.subheader("Predicted Race Data")
    st.table(pd.DataFrame([predicted_data]))

    # Update Driver Standings with Original Lap Times
    if not lap_times_df.empty:
        # Merge lap times with standings data to show original lap time for each driver
        avg_lap_times = lap_times_df.groupby('Driver')['Lap Time (s)'].mean().reset_index()
        standings_df = pd.merge(standings_df, avg_lap_times, on='Driver', how='left')

    # Keep the original points column intact
    standings_df = standings_df[['Driver', 'Position', 'Points']]  # Only keep Driver, Position, Points

    # Highlight the selected driver in the standings table
    def highlight_selected_racer(row):
        return ['background-color: yellow' if row['Driver'] == driver else '' for _ in row]

    # Display Standings Table with Selected Racer Highlighted
    st.subheader("Driver Standings")

    # Use st.dataframe for better customization and larger display
    st.dataframe(
        standings_df.style.apply(highlight_selected_racer, axis=1)
        .set_table_styles([
            {'selector': 'th', 'props': [('font-size', '16px')]},
            {'selector': 'td', 'props': [('font-size', '14px')]},
            # Increase width for the 'Points' column
            {'selector': '.col2', 'props': [('width', '150px')]},  # Points column (column index 2)
        ]),
        height=600
    )

    # New Lap Time Trend Graph
    if not lap_times_df.empty:
        # Create a line graph of lap times vs lap number for the selected driver
        driver_lap_times = lap_times_df[lap_times_df['Driver'] == driver]
        
        if not driver_lap_times.empty:
            lap_time_fig = px.line(
                driver_lap_times, x="Lap", y="Lap Time (s)", title=f"Lap Time Trend for {driver}",
                labels={"Lap": "Lap Number", "Lap Time (s)": "Lap Time (Seconds)"}
            )
            st.plotly_chart(lap_time_fig)

    # Training data visualization (for demonstration purposes)
    st.subheader("Training Data Visualization")
    fig = px.scatter(
        x=dummy_X[:, 0], y=dummy_y, labels={"x": "Lap Time (s)", "y": "Position"},
        title="Training Data: Lap Time vs Position"
    )
    st.plotly_chart(fig)

else:
    st.error("No standings data available for the selected race.")
