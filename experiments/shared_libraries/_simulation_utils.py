
from typing import Literal, Dict, List, TypedDict

from sklearn.base import BaseEstimator, RegressorMixin

import shared_libraries.data_processing as processing
import shared_libraries.algorithms as algorithms

import pandas as pd
import itertools
import statistics



CompoundType = Literal["SOFT", "MEDIUM", "HARD"]
RealCompoundType = Literal["C1", "C2", "C3", "C4", "C5", "C6"]

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

class StrategyDataWithoutWeatherAndCompoundMapping(TypedDict):
    Laps: pd.DataFrame
    Strategies: pd.DataFrame

class StrategyData(TypedDict):
    Laps: pd.DataFrame
    Strategies: pd.DataFrame

class __Lap(TypedDict):
    LapNumber: int
    TyreLife: int
    Compound: CompoundType
    IsPitLap: bool

class StrategyDataPostEvaluation(StrategyData):
    pass

class __Strategy:
    def __init__(self, initial_compound: CompoundType, stops: Dict[int, CompoundType]) -> None:
        self.initial_compound: CompoundType = initial_compound
        self.stops = stops

    def __str__(self) -> str:
        return f"Strategy[{[self.initial_compound] + list(self.stops.values())}, {list(self.stops.keys())}]"
    
    def __repr__(self) -> str:
        return str(self)
    
def evaluate_strategies(strategy_data: StrategyData, model) -> StrategyDataPostEvaluation:
    strategies_df = strategy_data["Strategies"].convert_dtypes()
    laps_df: pd.DataFrame = strategy_data["Laps"].copy()
    ml_part = laps_df.drop("StrategyId", axis="columns")

    laps_df["LapTimeZScore"] = model.predict(ml_part)
    
    mean_z_scores = (
        laps_df
            .loc[:, ["StrategyId", "LapTimeZScore"]]
            .groupby(by="StrategyId")
            .mean()
            ["LapTimeZScore"]
    )

    strategies_df["MeanZScore"] = mean_z_scores
    
    return {
        "Strategies": strategies_df,
        "Laps": laps_df
    }

def prepare_strategy_data_without_weather_and_weather_context(
            race_length: int, 
            min_stint_length: int,
            max_stint_length: int
        ) -> StrategyDataWithoutWeatherAndCompoundMapping:
    strategies = _prepare_strategies(race_length, min_stint_length, max_stint_length)
    strategies_df = _create_strategy_dataframe(strategies)
    lap_data = _prepare_lap_data(strategies_df, race_length)

    return {
        "Strategies": strategies_df,
        "Laps": lap_data
    }

def prepare_full_strategy_data(
            basic_strategy_data: StrategyDataWithoutWeatherAndCompoundMapping,
            compound_mapping: CompoundMapping,
            weather: Weather
        ) -> StrategyData:
    strategies_df = basic_strategy_data["Strategies"]
    lap_data = basic_strategy_data["Laps"]
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


compound_types = ["SOFT", "MEDIUM", "HARD"]
def _prepare_strategies(race_length: int, min_stint_length: int, max_stint_length: int) -> List[__Strategy]:
    if min_stint_length < 1:
        raise RuntimeError("min_stint_length cannot be less than 1.")
    if max_stint_length < min_stint_length:
        raise RuntimeError("max_stint_length must not be less than min_stint_length.")
    last_possible_stop = race_length - min_stint_length
    if min_stint_length > last_possible_stop:
        raise RuntimeError("min_stint_length is too long relative to race_length for any stops to occur.")
    
    min_stops = max((race_length - (max_stint_length - 1)) // max_stint_length, 0)
    max_stops = max(((race_length - (min_stint_length - 1))) // min_stint_length, 0)

    all_possible_stop_plans = []
    possible_compound_combinations = []
    for stop_count in range(min_stops, max_stops + 1):
        all_possible_stop_plans.append(algorithms.spaced_combinations(stop_count, race_length, min_stint_length, max_stint_length))
        combinations =  itertools.combinations_with_replacement(compound_types, stop_count + 1)
        combinations_with_at_least_two_compound_types = filter(lambda c: len(set(c)) >= 2, combinations)
        possible_compound_combinations.append(list(combinations_with_at_least_two_compound_types))

    strategies = []
    for stop_plans, compound_combinations in zip(all_possible_stop_plans, possible_compound_combinations):
        for plan in stop_plans:
            for compounds in compound_combinations:
                initial_compound = compounds[0]
                pit_compounds = compounds[1:]
                pit_stops = {}
                for compound, stop in zip(pit_compounds, plan):
                    pit_stops[stop] = compound
                strategies.append(__Strategy(initial_compound, pit_stops))

    return strategies


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


