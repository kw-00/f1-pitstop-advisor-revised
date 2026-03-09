import shared_libraries._simulation_utils as utils

from typing import List, Dict, TypedDict
import numpy as np
import pandas as pd



def prepare_example_weather_data() -> Dict[str, utils.Weather]:
    return {
        "Desert": {
            "AirTemp": 33,        
            "Humidity": 28,       
            "Pressure": 1009,     
            "Rainfall": False,
            "TrackTemp": 42,      
            "WindSpeed": 8,       
            "WindDirection": "NW"
        },
        "Tropical": {
            "AirTemp": 30,        
            "Humidity": 85,       
            "Pressure": 1007,
            "Rainfall": False,    
            "TrackTemp": 34,
            "WindSpeed": 4,
            "WindDirection": "SE"
        },
        # Temperate climate
        "Temperate": {
            "AirTemp": 20,        
            "Humidity": 70,       
            "Pressure": 1015,
            "Rainfall": False,     
            "TrackTemp": 22,      
            "WindSpeed": 10,
            "WindDirection": "SW"
        }
    }


def prepare_example_compound_mappins() -> Dict[str, utils.CompoundMapping]:
    return {
        "Fast": {
            "SOFT": "C1",
            "MEDIUM": "C2",
            "HARD": "C3"
        },
        "Medium": {
            "SOFT": "C2",
            "MEDIUM": "C3",
            "HARD": "C4"
        },
        "Durable": {
            "SOFT": "C3",
            "MEDIUM": "C4",
            "HARD": "C5"
        }
    }

class SimulationResults(TypedDict):
    WeatherAndMappingCombinations: pd.DataFrame
    FullStrategyData: Dict[int, utils.FullStrategyData]

def prepare_data_for_simulation(
    circuits: List[str],
    compound_mappings: Dict[str, utils.CompoundMapping],
    weathers: Dict[str, utils.Weather],
    circuit_lap_counts: pd.DataFrame
) -> SimulationResults:
    
    weather_and_mapping_combinations = []
    full_strategy_data = {}
    for circuit in circuits:
        lap_count = circuit_lap_counts.log[circuit, "LapCount"]
        basic_strategy_data = utils.prepare_strategy_data_without_weather_and_weather_context(lap_count, 15, 40)
        for weather_name, weather in weathers.items():
            for compound_mapping_name, compound_mapping in compound_mappings.items():
                id = len(weather_and_mapping_combinations)

                weather_and_mapping_combinations.append({
                    "Id": id, 
                    "WeatherName": weather_name, 
                    "CompoundMappingName": compound_mapping_name
                })

                full_strategy_data[id] = utils.prepare_full_strategy_data(
                    basic_strategy_data, 
                    compound_mapping, 
                    weather
                )

    return {
        "WeatherAndMappingCombinations": pd.DataFrame(weather_and_mapping_combinations),
        "FullStrategyData": full_strategy_data
    }


class EvaluationResults(TypedDict):
    EvaluationResults: pd.DataFrame

def evaluate_strategies_and_report(results: SimulationResults) 


