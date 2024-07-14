from dataclasses import dataclass
import os
import random

import joblib
import numpy as np
import pandas as pd
# ML models and utils
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score

from scripts.transform import TransformData


@dataclass
class SimulationParams:
    initial_capital: float
    threshold: float
    fees: float
    top_k: int
    portfolio_optimization: bool
    stop_loss: float
    take_profit: float
    lower_entry: float


class TrainModel:
    PREDICTION_NAME = 'pred_rf_best'
    PROBA_NAME = 'proba_pred'

    transformed_df: pd.DataFrame  # input dataframe from the Transformed piece
    df_full: pd.DataFrame  # full dataframe with DUMMIES

    # Dataframes for ML
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_valid_df: pd.DataFrame

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    X_train_valid: pd.DataFrame
    X_all: pd.DataFrame

    # feature sets
    GROWTH: list
    OHLCV: list
    CATEGORICAL: list
    TO_PREDICT: list
    TECHNICAL_INDICATORS: list
    TECHNICAL_PATTERNS: list
    MACRO: list
    NUMERICAL: list
    CUSTOM_NUMERICAL: list
    DUMMIES: list

    def __init__(self, transformed: TransformData):
        # init transformed_df
        self.transformed_df = transformed.transformed_df.copy(deep=True)
        self.transformed_df['ln_volume'] = self.transformed_df.Volume.apply(lambda x: np.log(x) if x > 0 else np.nan)
        # self.transformed_df['Date'] = pd.to_datetime(self.transformed_df['Date']).dt.strftime('%Y-%m-%d')

        self.model = None

    def _define_feature_sets(self):
        self.GROWTH = [g for g in self.transformed_df if (g.find('growth_') == 0) & (g.find('future') < 0)]
        self.OHLCV = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
        self.TO_PREDICT = [g for g in self.transformed_df.keys() if (g.find('future') >= 0)]
        self.MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
                      'DGS1', 'DGS5', 'DGS10']
        self.CUSTOM_NUMERICAL = ['vix_adj_close', 'SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative',
                                 'volatility', 'ln_volume']

        # artifacts from joins and/or unused original vars
        self.TO_DROP = ['Year', 'Date', 'Month_x', 'Month_y', 'index', 'Quarter', 'index_x',
                        'index_y'] + self.CATEGORICAL + self.OHLCV

        # All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
        self.TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc',
                                     'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
                                     'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
                                     'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
                                     'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
                                     'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
                                     'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
                                     'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
                                     'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']
        self.TECHNICAL_PATTERNS = [g for g in self.transformed_df.keys() if g.find('cdl') >= 0]

        self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + \
                         self.CUSTOM_NUMERICAL + self.MACRO

        # CHECK: NO OTHER INDICATORS LEFT
        self.OTHER = [k for k in self.transformed_df.keys() if
                      k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT]
        return

    def _define_dummies(self):
        # dummy variables can't be generated from Date and numeric variables ==> convert to STRING (to define groups for Dummies)
        # self.transformed_df.loc[:,'Month'] = self.transformed_df.Month_x.dt.strftime('%B')
        self.transformed_df.loc[:, 'Month'] = self.transformed_df.Month_x.astype(str)
        self.transformed_df['Weekday'] = self.transformed_df['Weekday'].astype(str)

        # define week of month
        self.transformed_df['wom'] = self.transformed_df["Date"].apply(lambda d: (d.day - 1) // 7 + 1)
        self.transformed_df.loc[:, 'wom'] = self.transformed_df.loc[:, 'wom'].astype(str)

        self.transformed_df.loc[:, 'month_wom'] = self.transformed_df.Month + '_w' + self.transformed_df.wom
        # del wom temp variable

        self.transformed_df.drop(columns=['wom'], inplace=True)

        # Generate dummy variables (no need for bool, let's have int32 instead)
        dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype='int32')
        self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
        # get dummies names in a list
        self.DUMMIES = dummy_variables.keys().to_list()

    def _perform_temporal_split(self, df: pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15,
                                test_prop=0.15):
        """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

    Args:
        df (DataFrame): The DataFrame to split.
        min_date (str or Timestamp): Minimum date in the DataFrame.
        max_date (str or Timestamp): Maximum date in the DataFrame.
        train_prop (float): Proportion of data for training set (default: 0.7).
        val_prop (float): Proportion of data for validation set (default: 0.15).
        test_prop (float): Proportion of data for test set (default: 0.15).

    Returns:
        DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
    """
        # Define the date intervals
        train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
        val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

        # Assign split labels based on date ranges
        split_labels = []
        for date in df['Date']:
            if date <= train_end:
                split_labels.append('train')
            elif date <= val_end:
                split_labels.append('validation')
            else:
                split_labels.append('test')

        # Add 'split' column to the DataFrame
        df['split'] = split_labels

        return df

    def _define_dataframes_for_ML(self):

        features_list = self.NUMERICAL + self.DUMMIES
        # What we're trying to predict?
        to_predict = 'is_positive_growth_5d_future'

        self.train_df = self.df_full[self.df_full.split.isin(['train'])].copy(deep=True)
        self.valid_df = self.df_full[self.df_full.split.isin(['validation'])].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full.split.isin(['train', 'validation'])].copy(deep=True)
        self.test_df = self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)

        # Separate numerical features and target variable for training and testing sets
        self.X_train = self.train_df[features_list + [to_predict]]
        self.X_valid = self.valid_df[features_list + [to_predict]]
        self.X_train_valid = self.train_valid_df[features_list + [to_predict]]
        self.X_test = self.test_df[features_list + [to_predict]]
        # this to be used for predictions and join to the original dataframe new_df
        self.X_all = self.df_full[features_list + [to_predict]].copy(deep=True)

        # Clean from +-inf and NaNs:

        self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
        self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
        self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
        self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
        self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)

        self.y_train = self.X_train[to_predict]
        self.y_valid = self.X_valid[to_predict]
        self.y_train_valid = self.X_train_valid[to_predict]
        self.y_test = self.X_test[to_predict]
        self.y_all = self.X_all[to_predict]

        # remove y_train, y_test from X_ dataframes
        del self.X_train[to_predict]
        del self.X_valid[to_predict]
        del self.X_train_valid[to_predict]
        del self.X_test[to_predict]
        del self.X_all[to_predict]

        print(f'length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}')
        print(f'  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}')

    def _clean_dataframe_from_inf_and_nan(self, df: pd.DataFrame):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def prepare_dataframe(self):
        print("Prepare the dataframe: define feature sets, add dummies, temporal split")
        self._define_feature_sets()
        # get dummies and df_full
        self._define_dummies()

        # temporal split
        min_date_df = self.df_full.Date.min()
        max_date_df = self.df_full.Date.max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        # define dataframes for ML
        self._define_dataframes_for_ML()

        return

    def train_random_forest(self, max_depth=17, n_estimators=200):
        # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
        print('Training the best model (RandomForest (max_depth=18, n_estimators=500))')
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            random_state=42,
                                            n_jobs=-1)

        self.model = self.model.fit(self.X_train_valid, self.y_train_valid)

    def persist(self, data_dir: str):
        '''Save dataframes to files in a local directory 'dir' '''
        os.makedirs(data_dir, exist_ok=True)

        # Save the model to a file
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir, model_filename)
        joblib.dump(self.model, path)

    def load(self, data_dir: str):
        """Load files from the local directory"""
        os.makedirs(data_dir, exist_ok=True)
        # Save the model to a file
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir, model_filename)

        self.model = joblib.load(path)

    def make_inference(self, sim_params: SimulationParams):
        """Make inference according to already derived parameters of the best predictor"""

        # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
        print('Making inference')

        y_pred_all = self.model.predict_proba(self.X_all)
        # list of predictions for class "1"
        y_pred_all_class1 = [k[1] for k in y_pred_all]
        # (Numpy Array) np.array of predictions for class "1" , converted from a list
        y_pred_all_class1_array = np.array(
            y_pred_all_class1
        )

        self.df_full[self.PROBA_NAME] = y_pred_all_class1_array
        self.df_full[self.PREDICTION_NAME] = (y_pred_all_class1_array >= sim_params.threshold).astype(int)
        # define rank of the prediction - only for predicted stocks
        self.df_full[f"{self.PREDICTION_NAME}_rank"] = self.df_full.groupby("Date")[self.PREDICTION_NAME].rank(
            method="first",
            ascending=False
        )

    @staticmethod
    def rolling_max_min(df: pd.DataFrame, window=5):
        """Get the rolling max High and min Low for the next 5 days

        Sort the DataFrame: Sorting by Ticker and Date ensures that we are looking at each ticker's data
        in chronological order.
        Rolling window calculation: We use the rolling method with a window of 5 to calculate the maximum high
        and minimum low over the next 5 days.
        The shift method is used to align these values correctly with the current row.
        """
        # high/low in 5 days
        df['Max_High_Next_5'] = df['High'].rolling(window=window, min_periods=1).max().shift(-window + 1)
        df['Min_Low_Next_5'] = df['Low'].rolling(window=window, min_periods=1).min().shift(-window + 1)

        # low in 1 day (for lower entry)
        df['Min_Low_Next_1'] = df['Low'].rolling(window=1, min_periods=1).min().shift(-1)
        return df

    def extend_df_for_simulation(self):
        """Build a DataFrame for simulation"""

        # Apply the function to each group of Ticker
        # Important!: need to drop index from groupby operation (reset_index) - so that you can merge that afterwards
        result = self.df_full[['Date', 'High', 'Low', 'Adj Close', 'Ticker']].groupby('Ticker').apply(
            self.rolling_max_min).reset_index(drop=True)

        # Calculate the ratios + safe divide
        result['Ratio_MaxHighNext5_to_AdjClose'] = np.where(result['Adj Close'] == 0, np.nan,
                                                            result['Max_High_Next_5'] / result['Adj Close'])
        result['Ratio_MinLowNext5_to_AdjClose'] = np.where(result['Adj Close'] == 0, np.nan,
                                                           result['Min_Low_Next_5'] / result['Adj Close'])

        result['Ratio_MinLowNext1_to_AdjClose'] = np.where(result['Adj Close'] == 0, np.nan,
                                                           result['Min_Low_Next_1'] / result['Adj Close'])

        # Merge the results back with the original dataframe
        self.df_full = self.df_full.merge(
            result[['Date', 'Ticker', 'Max_High_Next_5', 'Min_Low_Next_5',
                    'Ratio_MaxHighNext5_to_AdjClose', 'Ratio_MinLowNext5_to_AdjClose',
                    'Ratio_MinLowNext1_to_AdjClose']],
            on=['Date', 'Ticker']
        )

    @staticmethod
    def get_fees(row, sim_params: SimulationParams):
        """fees, depending on lower_entry, take_profit, stop_loss"""
        if row['lower_entry'] == 0:  # no trade ==> no fees
            return 0

        # pay fees in all other cases
        return -row['investment'] * sim_params.fees

    @staticmethod
    def get_future_gross_return(row, sim_params: SimulationParams):
        """future_gross_return, depending on lower_entry, take_profit, stop_loss"""
        if row['lower_entry'] == 0:  # no trade, investment is untouched, no fees
            return row['investment']

        # buy trade is filled for ALL next cases:
        if row['take_profit'] == 1 and row['stop_loss'] == 1:
            if random.random() > 0.5:  # assume take_profit event was first
                return row['investment'] * (sim_params.take_profit + (1 - sim_params.lower_entry))
            else:  # assume stop_loss event was first
                return row['investment'] * (sim_params.stop_loss + (1 - sim_params.lower_entry))

        if row['take_profit'] == 1:  # take some good profit, pay fees
            return row['investment'] * (sim_params.take_profit + (1 - sim_params.lower_entry))

        if row['stop_loss'] == 1:  # fix the loss, pay fees
            return row['investment'] * (sim_params.stop_loss + (1 - sim_params.lower_entry))

        # no stop_loss and no take_profit
        if pd.isna(row['growth_future_5d']):
            return row['investment']  # no information on growth in 5 days --> return the same investment in 5 days
        else:
            return row['investment'] * (row['growth_future_5d'] + (1 - sim_params.lower_entry))

    def one_date_simulation(
        self,
        date: str,
        invest_sum: float,
        sim_params: SimulationParams,
    ):
        predictor = self.PROBA_NAME

        # 1. get TOP_K (or ALL) predictions from the predictor, that are higher than THE THRESHOLD
        if sim_params.top_k is None:
            one_day_predictions_df = self.df_full[
                (self.df_full.Date == date) & (self.df_full[predictor] > sim_params.threshold)]
        else:
            one_day_predictions_df = self.df_full[
                (self.df_full.Date == date)
                & (self.df_full[predictor] > sim_params.threshold)
                & (self.df_full[f"{self.PREDICTION_NAME}_rank"] <= sim_params.top_k)
            ]

        FIELDS = ['Adj Close', 'Ticker', 'Date', predictor, f"{self.PREDICTION_NAME}_rank", 'growth_future_5d',
                  'Ratio_MaxHighNext5_to_AdjClose', 'Ratio_MinLowNext5_to_AdjClose', 'Ratio_MinLowNext1_to_AdjClose']
        result_df = one_day_predictions_df[FIELDS].copy()

        # 2. Get non-normalized weights: probability-threshold + 0.01
        result_df['weight'] = result_df[predictor] - sim_params.threshold + 0.01

        # 3. Get normalized weights
        result_df['weight_norm'] = result_df['weight'] / result_df['weight'].sum()

        # 4. Make bets to allocate 'invest_sum' across all suitable predictions
        result_df['investment'] = result_df['weight_norm'] * invest_sum

        # 5. Lower Entry: the trade is executed only is Low price for next day is lower than the bet (Adj_Close_today * sim_params.lower_entry)
        # [ONLY TRADES with lower_entry==1 are filled by the exchange]
        result_df['lower_entry'] = (result_df['Ratio_MinLowNext1_to_AdjClose'] <= sim_params.lower_entry).astype(int)

        # 6. Stop Loss: happens if the current price (or Low price) goes below stop loss threshold during one of the next 5 periods (1 week)
        result_df['stop_loss'] = (result_df['Ratio_MinLowNext5_to_AdjClose'] <= sim_params.stop_loss).astype(int)

        # 7. Take Profit: take the money if the current Price (or Max_price) goes higher than sim_params.take_profit
        result_df['take_profit'] = (result_df['Ratio_MaxHighNext5_to_AdjClose'] >= sim_params.take_profit).astype(int)

        # 8. Calculate future returns (when the order is executed + stop_loss True/False + take_profit True/False)
        result_df['future_gross_return'] = result_df.apply(
            lambda row: self.get_future_gross_return(row, sim_params=sim_params), axis=1)
        result_df['fees'] = result_df.apply(lambda row: self.get_fees(row, sim_params=sim_params), axis=1)
        result_df['future_net_return'] = result_df['future_gross_return'] + result_df['fees']

        return result_df

    def simulate(self, params: SimulationParams):
        simulation_df = None

        # all dates for simulation
        all_dates = self.df_full[self.df_full.split == 'test'].sort_values(by='Date').Date.unique()

        # arrays of dates and capital available (capital for the first 5 days)
        dates = []
        # first 5 periods trade with 1/5 of the initial_capital. e.g. [2000,2000,2000,2000,2000]
        capital = 5 * [params.initial_capital / 5]

        self.extend_df_for_simulation()

        # growth_future_5d is not defined for the last 5 days : ALL, but last 5 dates
        for current_date in all_dates[0:-5]:
            # take the value or everything that you can sell from 5 days ago
            current_invest_sum = capital[-5]

            # one day simulation result
            one_day_simulation_results = self.one_date_simulation(
                date=current_date,
                invest_sum=current_invest_sum,
                sim_params=params,
            )

            # add capital available in 5 days
            if len(one_day_simulation_results) == 0:  # no predictions -> no trades
                capital.append(current_invest_sum)
            else:
                capital.append(one_day_simulation_results.future_net_return.sum())
            dates.append(current_date)

            if simulation_df is None:
                simulation_df = one_day_simulation_results
            else:
                simulation_df = pd.concat([simulation_df, one_day_simulation_results], ignore_index=True)

        # add last 5 days to make the count of data points equal for dates/capital arrays
        dates.extend(all_dates[-5:])
        capital_df = pd.DataFrame({'capital': capital}, index=pd.to_datetime(dates))

        # results:
        print(f'============================================================================================')
        print(f'SIMULATION STARTED')
        print(f'Simulations params: {params}')
        print(
            f' Count bids {len(simulation_df)} in total, avg.bids per day {len(simulation_df) / simulation_df.Date.nunique()},  filled bids {len(simulation_df[simulation_df.lower_entry == 1])}, fill bids percent = {len(simulation_df[simulation_df.lower_entry == 1]) / len(simulation_df)}')
        stop_loss_filter = (simulation_df.lower_entry == 1) & (simulation_df.stop_loss == 1)
        print(
            f'  Stop loss events: count = {len(simulation_df[stop_loss_filter])}, net loss = {simulation_df[stop_loss_filter].future_net_return.sum() - simulation_df[stop_loss_filter].investment.sum()} ')
        take_profit_filter = (simulation_df.lower_entry == 1) & (simulation_df.take_profit == 1)
        print(
            f'  Take profit events: count = {len(simulation_df[take_profit_filter])}, net profit = {simulation_df[take_profit_filter].future_net_return.sum() - simulation_df[take_profit_filter].investment.sum()} ')
        print(f'  Start capital = {params.initial_capital}, Resulting capital: {capital_df[-5:].capital.sum()} ')
        print(
            f'  CAGR in 4 years: {np.round((capital_df[-5:].capital.sum() / params.initial_capital) ** (1 / 4), 3)} or {np.round(((capital_df[-5:].capital.sum() / params.initial_capital) ** (1 / 4) - 1) * 100.0, 2)} % of avg. growth per year')
        print(f'============================================================================================')

        return simulation_df, capital_df

    def get_last_signals(self, num: int = 10):
        """Get the last num signals"""
        MIN_RANK = 8

        COLUMNS = ['Adj Close', 'Ticker', 'Date', self.PREDICTION_NAME, self.PREDICTION_NAME + '_rank']
        signals = self.df_full[(self.df_full[f"{self.PREDICTION_NAME}_rank"] <= MIN_RANK)
                        & (self.df_full[self.PREDICTION_NAME] > 0)].sort_values(
            by=["Date", f"{self.PREDICTION_NAME}_rank"])

        return signals[COLUMNS]
