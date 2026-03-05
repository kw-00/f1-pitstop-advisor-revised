from __future__ import annotations

import sys
sys.path.append("..")


import experiments.shared_libraries._data_processing_utils as processing
import pandas as pd
import pickle

from typing import Dict, List, Tuple
from fastf1.core import Session




def get_data_by_circuit(sessions: List[Session], compound_map: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sessions_and_compound_mappings = _join_sessions_and_compound_mappings(sessions, compound_map)

    sessions_and_lap_data: List[Tuple[Session, pd.DataFrame]] = []
    for session, compound_map in sessions_and_compound_mappings:
        mapping = compound_map.loc[["soft", "medium", "hard"]]
        lap_data = _get_lap_data(session)
        fitted_mapping = pd.concat([pd.DataFrame(mapping).T] * lap_data.shape[0], ignore_index=True)
        lap_data_with_compound_info = pd.concat([lap_data, fitted_mapping], axis="columns")
        sessions_and_lap_data.append((session, lap_data_with_compound_info))

    data_by_circuit_split: Dict[str, List[pd.DataFrame]] = {}

    for session, data in sessions_and_lap_data:
        circuit = session.session_info["Meeting"]["Circuit"]["ShortName"]
        if circuit not in data_by_circuit_split:
            data_by_circuit_split[circuit] = []
        data_by_circuit_split[circuit].append(data)

    data_by_circuit: Dict[str, pd.DataFrame] = {}
    for circuit, dfs in data_by_circuit_split.items():
        data_by_circuit[circuit] = pd.concat(dfs, axis="index", ignore_index=True)

    return data_by_circuit

def remove_first_laps_with_pit_stop(data: pd.DataFrame) -> None:
    data.drop(data[(data["LapNumber"] == 1) & (data["IsPitLap"] == True)].index, inplace=True)

def remove_laps_affected_by_unexpected_events(data: pd.DataFrame) -> None:
    data.drop(data[~data["TrackStatus"].apply(lambda status: "1" in status)].index, inplace=True)

def remove_outliers(data: pd.DataFrame) -> None:
    Q1 = data["LapTimeZScore"].quantile(0.25)
    Q3 = data["LapTimeZScore"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data.drop(data[(data["LapTimeZScore"] < lower_bound) & (data["LapTimeZScore"] > upper_bound)].index, inplace=True)

def remove_missing_values(data: pd.DataFrame) -> None:
    selected_columns = [
        "LapTimeZScore",
        "IsPitLap",
        "Compound",
        "TyreLife",
        "FreshTyre",
        "LapNumber",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindSpeed",
        "WindDirection"
    ]
    data.dropna(subset=selected_columns, inplace=True)

def add_real_compound(data: pd.DataFrame) -> None:
    for idx in data.index:
        if data.loc[idx, "Compound"] in ("HARD", "MEDIUM", "SOFT"):
            data.loc[idx, "RealCompound"] = data.loc[idx, data.loc[idx, "Compound"].lower()] # type: ignore
        else:
            data.loc[idx, "RealCompound"] = data.loc[idx, "Compound"]

def make_wind_direction_categorical(data: pd.DataFrame) -> None:
    # Pack WindDirection into bins
    mapping = {
        0: "N",
        1: "NE",
        2: "E",
        3: "SE",
        4: "S",
        5: "SW",
        6: "W",
        7: "NW"
    }
    def get_wind_direction(degrees):
        cat = round(degrees / 45) % 8
        return mapping[cat]
    
    data["WindDirection"] = data["WindDirection"].apply(get_wind_direction)

def remove_special_compounds(data: pd.DataFrame) -> None:
    allowed_compounds = ["SOFT", "MEDIUM", "HARD"]
    data.drop(data[data["Compound"].isin(allowed_compounds) == False].index, axis="index", inplace=True)

def select_columns_for_ml(data: pd.DataFrame) -> None:
    selected_columns = [
        "LapTimeZScore",
        "IsPitLap",
        "Compound",
        "RealCompound",
        "TyreLife",
        "LapNumber",
        "AirTemp",
        "Humidity",
        "Pressure",
        "Rainfall",
        "TrackTemp",
        "WindSpeed",
        "WindDirection"
    ]
    data.drop([c for c in data.columns if c not in selected_columns], axis="columns", inplace=True)

def add_missing_dummy_columns(data: pd.DataFrame) -> None:
    columns = []
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        columns.append(compound)
    for real_compound in ["C1", "C2", "C3", "C4", "C5"]:
        columns.append(f"RealCompound_{real_compound}")
    for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        columns.append(f"WindDirection_{direction}")

    for column in columns:
        if column not in data.columns:
            data[column] = False

def _join_sessions_and_compound_mappings(sessions: List[Session], compounds_map: pd.DataFrame) -> List[Tuple[Session, pd.DataFrame]]:
    # Create queue of compound data
    compd_q = []
    for idx in compounds_map.index:
        compd_q.append(compounds_map.loc[idx, :])
    compd_q.reverse()

    # Merge data from queue with sessions
    sess_n_compds = []
    for s in sessions:
        s_info = s.session_info
        while True:
            compds = compd_q.pop()
            year_matches = compds["year"] == s_info["StartDate"].year
            name_matches = compds["gp"] == s_info["Meeting"]["Name"]
            if year_matches and name_matches:
                break
        sess_n_compds.append((s, compds))

    return sess_n_compds


def _get_lap_data(session: Session) -> pd.DataFrame:
    data = processing.get_lap_data_with_weather(session)
    processing.add_z_score_for_laps(data, inplace=True)
    processing.add_is_pit_lap(data, inplace=True)
    return data