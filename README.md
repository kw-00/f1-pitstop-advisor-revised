# F1-PITSTOP-ADVISOR

## What this is about
The goal of this project is to create a system capable of creating effective pit-stop strategies for Formula 1 races.

In this context, a pit-stop strategy comprises two elements:
1. Pit-stop timing — which laps to take pit-stops on.
1. Tyre compound choice — what tyre compound do we start the race with, and what tyre compounds do we choose during each pit-stop.

## Software architecture
At its core, the *pit-stop advisor* consists of two parts — a regression model and a *strategy finder*.

### The regression model
The regression model leverages historical race data to model the effect of various factors on lap time. Each data point represents one lap of a race. The target attribute is a measure of **lap time**.

The most important attributes (apart from the target) are **tyre compound**, **tyre mileage** as well as **whether a pit-stop occurred** during the lap.

The values of those 3 attributes depend solely on pit-stop strategy. Let's call them the **Big Three**. 

"*We have a strategy*" leads us to "*We know the values for every lap of the race*".

These 3 variables have a significant influence on lap time. This is important. Through them, we can **model the influence of pit-stop strategy on lap times**, and therefore the total time it takes to complete all laps of the race. Which in turn directly influences a driver's success.

### The strategy finder
The strategy finder uses the regression model in order to find the best strategy for any given race.

First, it generates a large array of possible pit-stop strategies. Many strategies are considered, including non-standard ones.

#### Pit-stop strategy => lap data
Next, for each strategy, lap data is prepared. The values of the Big Three attributes are calculated for each lap. The remaining attributes describe the weather, and can be determined from weather forecasts or simply use current weather.

#### Lap data => how good is the strategy?
Finally, the regression model is used to determine lap times for each strategy. Based on total predicted race time, the strategies with the lowest times are selected.

### The best strategy determined
In this way, we determine the best strategy. The accuracy of that choice depends on the regression model's accuracy. 

## Potential bias
Factors unrelated to pit-stops such as driver skill or technological gaps between teams introduce bias. For example, suppose a great driver with an exceptionally fast car uses a mediocre strategy. However, his lap times are nonetheless exceptionally good. This introduces a correlation between bad strategy and good race times.

In order to mitigate this, I normalized the target attribute (lap time) using the **Z-score** formula.

## Z-score to mitigate it
*Z-score* indicates the relative deviation of a value from other values in a sample. In the case of this project, it is the deviation of lap time from other lap times.

What we choose as our sample is of paramount importance. Here, each sample consists of laps completed by the same driver, during the same session (race).

That way, the effect of confounding factors is minimized. The target attribute is now a measure of lap time relative to remaining laps times in that same sample.

Therefore:
- Each lap is compared to other laps by the same driver in the same session — differences between drivers no longer introduce strong bias. Even the effect of changes in a driver's skill is mitigated, since a driver's skill does not change significantly throughout one session.
- Each driver uses one car during a session — therefore, the effect of technological differences is also mitigated.

# The data processing pipeline
Each part of the process is represented by a Jupyter Notebook.

## Create dataset
[Notebook link](./experiments/1_create_dataset.ipynb)

The first step involves loading and processing the data. Here's a quick overview:
1. Extract lap data and weather data from *FastF1* and merge them together
1. Remove outliers with special consideration for pit laps and the first lap of every race
1. Visualize the lap time distribution of the remaining data
1. Calculate lap time Z-score
1. Aggregate data by F1 circuit
1. Remove columns that are not going to be used for training regression models
1. Apply one-hot encoding for categorical columns

For more details, check the notebook itself.

## Dataset analysis
[Notebook link](./experiments/2_dataset_analysis.ipynb)

Here we perform some data analysis. We take a look at the data in its current form and visualize the correlations between the target and the remaining attributes.

# Model selection
[Notebook link](./experiments/3_model_selection.ipynb)

Here we train regression models using a broad set of machine learning algorithms. For each algorithm, we tweak various parameters to find the best ones.

Lastly, we compare the best models for each algorithm and compare them. The best one will be used for strategy evaluation.

# Simulation
[Notebook link](./experiments/4_simulation.ipynb)

Finally, we generate a broad range of strategies and evaluate them using the selected model. Strategies are selected for every circuit, with various weather conditions and tyre compound nominations.

The Z-scores of each lap's time are plotted for every strategy generated. This allows a better understanding of how the system is performing and whether its strategic choices are sensible.

# Data source
The main data source is the [FastF1](https://docs.fastf1.dev/index.html) Python library. Secondly, data about historical compound nominations was scraped off the Internet, primarily Wikipedia.

## Acknowledgement
This project is based on my thesis project, which I completed together with [quaspar33](https://github.com/quaspar33). It is a comprehensively revised version of the original, featuring three direct improvements:
- Removal of laps affected by unpredictable incidents as well as outliers
- Simplified code with lower-level operations abstracted away through functions
- Significant optimization of the algorithms used to generate and evaluate strategies

# Environment installation
In order to prepare your Python environment for the project:
1. Make sure you have Python 3.11 installed
1. Ensure you're in the project's root folder
1. Run the following commands: 
    #### Windows

    ```py -3.11 -m venv .venv; .venv/Scripts/activate; pip install poetry; poetry install```

    #### Mac
    ```python3.11 -m venv .venv; source .venv/bin/activate; pip install poetry; poetry install```

