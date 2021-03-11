import abc
import numpy as np
import pandas as pd
import copy
import ast
from qlib.log import TimeInspector
from datetime import datetime
from qlib.data.dataset.utils import fetch_df_by_index
from qlib.utils.serial import Serializable
from qlib.utils.paral import datetime_groupby_apply

EPS = 1e-12

class Processor(Serializable):
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters
        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.
        """
        pass

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data
        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside
        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """
        pass

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.
        Returns
        -------
        bool:
            if it is usable for infenrece.
        """
        return True


class TanhProcess(Processor):
    """ Use tanh to process noise data"""

    def __call__(self, df):
        def tanh_denoise(data):
            mask = data.columns.get_level_values(1).str.contains("LABEL")
            col = df.columns[~mask]
            data[col] = data[col] - 1
            data[col] = np.tanh(data[col])

            return data

        return tanh_denoise(df)

class ProcessInf(Processor):
    """Process infinity  """
    def __call__(self, df):
        def replace_inf(data):
            def process_inf(df):
                for col in df.columns:
                    # FIXME: Such behavior is very weird
                    df[col] = df[col].replace([np.inf, -np.inf], df[col][~np.isinf(df[col])].mean())
                return df

            data = datetime_groupby_apply(data, process_inf)
            data.sort_index(inplace=True)
            return data

        return replace_inf(df)


class Fillna(Processor):
    """Process NaN"""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)
            df.fillna({col: self.fill_value for col in cols}, inplace=True)
        return df

class FormatLevelTwo(Processor):
    def __call__(self,df):
        def extract(row):
            output = {}
            for d in row.values:
                if str(type(d)) == "<class 'str'>":
                    d = ast.literal_eval(d)
                for k in d.keys():
                    output[k] = d[k]
            return output
        def convert(timestamp):
            if timestamp != 0:
                timestamp = timestamp/1000
                newtime = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %I:%M:%S")
            else:
                newtime = np.nan
            return newtime
        def process_timestamps(df):
            df.bidtimestamp = df.bidtimestamp.apply(convert)
            df.asktimestamp = df.asktimestamp.apply(convert)
            return df
        def process_leveltwo(df):
            df = df[pd.notnull(df['0'])]
            df.columns = ["index","dict"]
            df = df.groupby('index')['dict'].apply(extract).unstack().fillna(0)
            bid_ask = df.groupby("index")["data"].apply(extract).unstack().fillna(0)
            df = df.join(bid_ask).drop("data",1)
            tempdf = df.copy()
            x = [{'price': np.nan, 'size': np.nan, 'timestamp': np.nan}]
            tempdf['bids'] = df.bids.apply(lambda y: x if len(y)==0 else y)
            tempdf['asks'] = df.asks.apply(lambda y: x if len(y)==0 else y)
            tempdf['bids'] = tempdf['bids'].str[0]
            tempdf['asks'] = tempdf['asks'].str[0]
            bidinfo = tempdf.groupby("index")["bids"].apply(extract).unstack().fillna(0)
            bidinfo.columns = ["bidprice","bidsize", "bidtimestamp"]
            askinfo = tempdf.groupby("index")["asks"].apply(extract).unstack().fillna(0)
            askinfo.columns = ["askprice","asksize", "asktimestamp"]
            tempdf = tempdf.join(bidinfo).join(askinfo).drop("bids",1).drop("asks",1)
            process_timestamps(tempdf)
            df = tempdf.copy()
            return(tempdf)
        return process_leveltwo(df)
