
from typing import Literal, Dict, List, TypedDict

import shared_libraries.data_processing as processing

import pandas as pd



CompoundType = Literal["SOFT", "MEDIUM", "HARD"]
RealCompoundType = Literal["C1", "C2", "C3", "C4", "C5"]

class CompoundMapping(TypedDict):
    SOFT: RealCompoundType
    MEDIUM: RealCompoundType
    HARD: RealCompoundType

WindDirectionType = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

class Weather(TypedDict):
    AirTemp: float
    Humidity: int
    Pressure: float
    Rainfall: bool
    TrackTemp: float
    WindSpeed: float
    WindDirection: WindDirectionType

class StrategyData(TypedDict):
    Laps: pd.DataFrame
    Strategies: pd.DataFrame

class __Lap(TypedDict):
    LapNumber: int
    TyreLife: int
    Compound: CompoundType
    IsPitLap: bool

class __Strategy:
    def __init__(self, initial_compound: CompoundType, stops: Dict[int, CompoundType]) -> None:
        self.initial_compound: CompoundType = initial_compound
        self.stops = stops

def prepare_strategy_data(
            compound_mapping: CompoundMapping,
            weather: Weather, 
            race_length: int, 
            min_stint_length: int,
            max_stint_length: int
        ) -> StrategyData:
    strategies = _prepare_strategies(race_length, min_stint_length, max_stint_length)
    strategies_df = _create_strategy_dataframe(strategies)
    lap_data = _prepare_lap_data(strategies_df, race_length)
    real_compounds = _get_real_compounds(lap_data, compound_mapping)
    weather_df = _get_weather_dataframe(lap_data, weather)
    full_lap_data = pd.concat([lap_data, weather_df], axis="columns")
    full_lap_data["RealCompound"] = real_compounds
    dummies = pd.get_dummies(full_lap_data)
    processing.add_missing_dummy_columns(dummies)
    return {
        "Strategies": strategies_df,
        "Laps": dummies
    }


def _prepare_strategies(race_length: int, min_stint_length: int, max_stint_length: int) -> List[__Strategy]:
    # TODO - strategy generation
    return [
        __Strategy("SOFT", {20: "MEDIUM"}),
        __Strategy("MEDIUM", {25: "HARD"})
    ]

def _create_strategy_dataframe(strategies: List[__Strategy]) -> pd.DataFrame:
    return pd.DataFrame({
        "Id": list(range(len(strategies))),
        "Strategy": strategies
    })

def _prepare_lap_data(strategy_dataframe: pd.DataFrame, race_length: int) -> pd.DataFrame:
    race_dfs = []
    for idx in strategy_dataframe["Id"].array:
        race_laps = _get_race(strategy_dataframe.loc[idx, "Strategy"], race_length) # type: ignore
        race_df = pd.concat(
            [
                pd.DataFrame({
                    "StrategyId": [idx] * len(race_laps)
                }),
                pd.DataFrame(race_laps)
            ]
        , axis="columns")
        race_dfs.append(race_df)
    return pd.concat(race_dfs, axis="index", ignore_index=True)
    

def _get_real_compounds(lap_data: pd.DataFrame, compound_mapping: CompoundMapping) -> pd.Series:
    return lap_data["Compound"].map(lambda compound: compound_mapping[compound])

def _get_weather_dataframe(lap_data: pd.DataFrame, weather: Weather) -> pd.DataFrame:
    return pd.DataFrame([weather] * lap_data.shape[0])




def _get_race(strategy: __Strategy, race_length: int) -> List[__Lap]:
    laps: List[__Lap] = []
    compound = strategy.initial_compound
    tyre_life = 1
    for lap_number in range(1, race_length + 1):
        if lap_number in strategy.stops:
            is_pit_lap = True
        else:
            is_pit_lap = False

        laps.append({
            "LapNumber": lap_number,
            "TyreLife": tyre_life,
            "Compound": compound,
            "IsPitLap": is_pit_lap
        })

        if is_pit_lap:
            compound = strategy.stops[lap_number]
            tyre_life = 1
        else:
            tyre_life += 1


    return laps


