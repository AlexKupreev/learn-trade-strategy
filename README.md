# Learning Trading Strategy

A capstone project for the [Stock Markets Analytics Zoomcamp 2024](https://courses.datatalks.club/sma-zoomcamp-2024/).

## The goal

Implement a base trading strategy for top US, EU, Chinese and Indian stocks.


## Implementation

Implementation is strongly based on the [code from learning materials](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/tree/main) 
with addition of 10 Chinese stocks and some changes in prediction cases. 

## Development and running the logic

### Local development/running

* Clone the repository
* Ensure pipenv is installed locally
* Ensure that TA-lib is installed as described in the [docs](https://github.com/TA-Lib/ta-lib-python?tab=readme-ov-file#dependencies)
* Run `pipenv shell` to activate the virtual environment
* Run `pipenv install --dev` to install the dependencies along with Jupyter and dev utilities (ruff linter/formatter)
* Running using Jupyter
  * Connect to local Jupyter server (look into your IDE docs)
  * Go to the [notebook](notebooks/workflow.ipynb), fill in necessary settings and run all steps 
* Running from command line
  * Fill in configuration in the [config](config.toml) file (name may be changed, see CLI help)
  * Run the script
  ```
   # default run, use pre-fetched data and pre-trained model for inference and simulation
   $ python main.py
   # "full" run - with getting data from API, transforming, training the model, making inference and simulation
   $ python main.py --fetch-repo --transform-data --train-model
   # CLI help
   $ python main.py --help
   ```

### Cloud notebooks

* [learn_trade_strategy_capstone_data](notebooks/learn_trade_strategy_capstone_data.ipynb) - collect the data and pack into a single dataframe/file
* [learn_trade_strategy_capstone_features_eda_model_trade](notebooks/learn_trade_strategy_capstone_features_eda_model_trade.ipynb) - feature engineering, EDA, model training and trading strategy implementation
