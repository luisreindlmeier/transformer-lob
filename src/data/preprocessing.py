import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from src import config as cfg


def reset_indexes(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    dataframes[0] = dataframes[0].reset_index(drop=True)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes


def z_score_orderbook(data: pd.DataFrame, mean_size: Optional[float] = None, mean_prices: Optional[float] = None,
                      std_size: Optional[float] = None, std_prices: Optional[float] = None
                      ) -> Tuple[pd.DataFrame, float, float, float, float]:
    if mean_size is None or std_size is None:
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    if mean_prices is None or std_prices is None:
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()

    for col in data.columns[1::2]:
        data[col] = (data[col].astype("float64") - mean_size) / std_size
    for col in data.columns[0::2]:
        data[col] = (data[col].astype("float64") - mean_prices) / std_prices

    return data, mean_size, mean_prices, std_size, std_prices


def normalize_messages(data: pd.DataFrame, mean_size: Optional[float] = None, mean_prices: Optional[float] = None,
                       std_size: Optional[float] = None, std_prices: Optional[float] = None,
                       mean_time: Optional[float] = None, std_time: Optional[float] = None,
                       mean_depth: Optional[float] = None, std_depth: Optional[float] = None
                       ) -> Tuple[pd.DataFrame, float, float, float, float, float, float, float, float]:
    if mean_size is None: mean_size = data["size"].mean()
    if std_size is None: std_size = data["size"].std()
    if mean_prices is None: mean_prices = data["price"].mean()
    if std_prices is None: std_prices = data["price"].std()
    if mean_time is None: mean_time = data["time"].mean()
    if std_time is None: std_time = data["time"].std()
    if mean_depth is None: mean_depth = data["depth"].mean()
    if std_depth is None: std_depth = data["depth"].std()

    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth

    data["event_type"] = data["event_type"] - 1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)

    return data, mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth


def labeling(X: np.ndarray, len_smooth: int, h: int) -> np.ndarray:
    if h < len_smooth:
        len_smooth = h
    
    previous_ask = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len_smooth)[:-h]
    previous_bid = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len_smooth)[:-h]
    future_ask = np.lib.stride_tricks.sliding_window_view(X[:, 0], window_shape=len_smooth)[h:]
    future_bid = np.lib.stride_tricks.sliding_window_view(X[:, 2], window_shape=len_smooth)[h:]

    previous_mid = np.mean((previous_ask + previous_bid) / 2, axis=1)
    future_mid = np.mean((future_ask + future_bid) / 2, axis=1)

    pct_change = (future_mid - previous_mid) / previous_mid
    alpha = np.abs(pct_change).mean() / 2

    return np.where(pct_change < -alpha, 2, np.where(pct_change > alpha, 0, 1))


class LOBSTERPreprocessor:
    COLUMNS_NAMES = {
        "orderbook": ["sell1", "vsell1", "buy1", "vbuy1", "sell2", "vsell2", "buy2", "vbuy2",
                      "sell3", "vsell3", "buy3", "vbuy3", "sell4", "vsell4", "buy4", "vbuy4",
                      "sell5", "vsell5", "buy5", "vbuy5", "sell6", "vsell6", "buy6", "vbuy6",
                      "sell7", "vsell7", "buy7", "vbuy7", "sell8", "vsell8", "buy8", "vbuy8",
                      "sell9", "vsell9", "buy9", "vbuy9", "sell10", "vsell10", "buy10", "vbuy10"],
        "message": ["time", "event_type", "order_id", "size", "price", "direction"]
    }
    
    def __init__(self, raw_data_dir: str, output_dir: str, split_rates: Optional[List[float]] = None) -> None:
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.split_rates = split_rates or cfg.SPLIT_RATES
        self.n_lob_levels = cfg.N_LOB_LEVELS
        self.dataframes: List[List[pd.DataFrame]] = []
    
    def preprocess(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')])
        self.num_trading_days = len(csv_files) // 2
        
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(split_days)
        
        for i in range(len(self.dataframes)):
            if len(self.dataframes[i][0]) == 0:
                continue
            self.dataframes[i][0]["price"] = self.dataframes[i][0]["price"].astype(float) / 10000
            price_cols = self.dataframes[i][1].columns[::2]
            self.dataframes[i][1][price_cols] = self.dataframes[i][1][price_cols].astype(float) / 10000
        
        train_input = self.dataframes[0][1].values
        val_input = self.dataframes[1][1].values
        test_input = self.dataframes[2][1].values
        
        for i, h in enumerate(cfg.LOBSTER_HORIZONS):
            train_labels = labeling(train_input, cfg.LEN_SMOOTH, h)
            val_labels = labeling(val_input, cfg.LEN_SMOOTH, h) if len(val_input) > 0 else np.array([])
            test_labels = labeling(test_input, cfg.LEN_SMOOTH, h)
            
            train_labels = np.concatenate([train_labels, np.full(train_input.shape[0] - len(train_labels), np.inf)])
            val_labels = np.concatenate([val_labels, np.full(val_input.shape[0] - len(val_labels), np.inf)]) if len(val_input) > 0 else np.array([])
            test_labels = np.concatenate([test_labels, np.full(test_input.shape[0] - len(test_labels), np.inf)])
            
            if i == 0:
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=[f"label_h{h}"])
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=[f"label_h{h}"]) if len(val_labels) > 0 else pd.DataFrame(columns=[f"label_h{h}"])
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=[f"label_h{h}"])
            else:
                self.train_labels_horizons[f"label_h{h}"] = train_labels
                if len(val_labels) > 0:
                    self.val_labels_horizons[f"label_h{h}"] = val_labels
                self.test_labels_horizons[f"label_h{h}"] = test_labels
        
        self._normalize_dataframes()
        self._save()
        print(f"Preprocessing complete. Output saved to {self.output_dir}")
    
    def _split_days(self) -> List[int]:
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"Train: {train} days, Val: {val - train} days, Test: {test - val} days")
        return [train, val, test]
    
    def _create_dataframes_splitted(self, split_days: List[int]) -> None:
        csv_files = sorted([f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')])
        train_messages = train_orderbooks = val_messages = val_orderbooks = test_messages = test_orderbooks = None
        
        for i, filename in enumerate(csv_files):
            filepath = os.path.join(self.raw_data_dir, filename)
            is_message = (i % 2) == 0
            
            if i < split_days[0]:
                if is_message:
                    if i == 0:
                        train_messages = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                    else:
                        train_message = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                else:
                    if i == 1:
                        train_orderbooks = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        train_orderbooks, train_messages = self._preprocess_message_orderbook([train_messages, train_orderbooks])
                    else:
                        train_orderbook = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        train_orderbook, train_message = self._preprocess_message_orderbook([train_message, train_orderbook])
                        train_messages = pd.concat([train_messages, train_message], axis=0)
                        train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)
            
            elif split_days[0] <= i < split_days[1]:
                if is_message:
                    if i == split_days[0]:
                        self.dataframes.append([train_messages, train_orderbooks])
                        val_messages = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                    else:
                        val_message = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                else:
                    if i == split_days[0] + 1:
                        val_orderbooks = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        val_orderbooks, val_messages = self._preprocess_message_orderbook([val_messages, val_orderbooks])
                    else:
                        val_orderbook = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        val_orderbook, val_message = self._preprocess_message_orderbook([val_message, val_orderbook])
                        val_messages = pd.concat([val_messages, val_message], axis=0)
                        val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)
            
            else:
                if is_message:
                    if i == split_days[1]:
                        if len(self.dataframes) == 0:
                            self.dataframes.append([train_messages, train_orderbooks])
                        if val_messages is None:
                            val_messages = pd.DataFrame(columns=self.COLUMNS_NAMES["message"])
                            val_orderbooks = pd.DataFrame(columns=self.COLUMNS_NAMES["orderbook"][:self.n_lob_levels * cfg.LEN_LEVEL])
                        self.dataframes.append([val_messages, val_orderbooks])
                        test_messages = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                    else:
                        test_message = pd.read_csv(filepath, names=self.COLUMNS_NAMES["message"])
                else:
                    if i == split_days[1] + 1:
                        test_orderbooks = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        test_orderbooks, test_messages = self._preprocess_message_orderbook([test_messages, test_orderbooks])
                    else:
                        test_orderbook = pd.read_csv(filepath, names=self.COLUMNS_NAMES["orderbook"])
                        test_orderbook, test_message = self._preprocess_message_orderbook([test_message, test_orderbook])
                        test_messages = pd.concat([test_messages, test_message], axis=0)
                        test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)
        
        self.dataframes.append([test_messages, test_orderbooks])
    
    def _preprocess_message_orderbook(self, dataframes: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        dataframes = reset_indexes(dataframes)
        min_len = min(len(dataframes[0]), len(dataframes[1]))
        dataframes[0] = dataframes[0].iloc[:min_len]
        dataframes[1] = dataframes[1].iloc[:min_len, :self.n_lob_levels * cfg.LEN_LEVEL]
        dataframes = reset_indexes(dataframes)
        
        indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
        valid_indexes = indexes_to_drop.intersection(dataframes[1].index)
        dataframes[0] = dataframes[0].drop(valid_indexes)
        dataframes[1] = dataframes[1].drop(valid_indexes)
        dataframes = reset_indexes(dataframes)
        
        dataframes[0] = dataframes[0].drop(columns=["order_id"])
        first_time = dataframes[0]["time"].values[0]
        dataframes[0]["time"] = dataframes[0]["time"].diff()
        dataframes[0].iat[0, dataframes[0].columns.get_loc("time")] = first_time - 34200
        dataframes[0]["depth"] = 0
        
        prices = dataframes[0]["price"].values
        directions = dataframes[0]["direction"].values
        event_types = dataframes[0]["event_type"].values
        bid_sides = dataframes[1].iloc[:, 2::4].values
        ask_sides = dataframes[1].iloc[:, 0::4].values
        
        depths = np.zeros(dataframes[0].shape[0], dtype=int)
        max_orderbook_idx = len(bid_sides) - 1
        for j in range(1, len(prices)):
            index = min(j if event_types[j] == 1 else j - 1, max_orderbook_idx)
            if directions[j] == 1:
                depth = (bid_sides[index, 0] - prices[j]) // 100
            else:
                depth = (prices[j] - ask_sides[index, 0]) // 100
            depths[j] = max(depth, 0)
        
        dataframes[0]["depth"] = depths
        dataframes[0] = dataframes[0].iloc[1:, :]
        dataframes[1] = dataframes[1].iloc[1:, :]
        dataframes = reset_indexes(dataframes)
        
        dataframes[0]["direction"] = dataframes[0]["direction"] * dataframes[0]["event_type"].apply(lambda x: -1 if x == 4 else 1)
        return dataframes[1], dataframes[0]
    
    def _normalize_dataframes(self) -> None:
        for i in range(len(self.dataframes)):
            if len(self.dataframes[i][1]) == 0:
                continue
            if i == 0:
                self.dataframes[i][1], mean_size_ob, mean_prices_ob, std_size_ob, std_prices_ob = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_size_ob, mean_prices_ob, std_size_ob, std_prices_ob)
        
        for i in range(len(self.dataframes)):
            if len(self.dataframes[i][0]) == 0:
                continue
            if i == 0:
                self.dataframes[i][0], mean_size_msg, mean_prices_msg, std_size_msg, std_prices_msg, mean_time, std_time, mean_depth, std_depth = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_size_msg, mean_prices_msg, std_size_msg, std_prices_msg, mean_time, std_time, mean_depth, std_depth)
    
    def _save(self) -> None:
        for i, name in enumerate(["train", "val", "test"]):
            messages, orderbook = self.dataframes[i]
            labels = [self.train_labels_horizons, self.val_labels_horizons, self.test_labels_horizons][i]
            if len(messages) == 0 or len(orderbook) == 0:
                combined = np.empty((0, 50))
                output_path = os.path.join(self.output_dir, f"{name}.npy")
                np.save(output_path, combined)
                print(f"  Saved {output_path}: shape {combined.shape} (empty)")
                continue
            messages = messages.reset_index(drop=True)
            orderbook = orderbook.reset_index(drop=True)
            input_data = pd.concat([messages, orderbook], axis=1)
            combined = pd.concat([pd.DataFrame(input_data.values), pd.DataFrame(labels.values)], axis=1).values
            output_path = os.path.join(self.output_dir, f"{name}.npy")
            np.save(output_path, combined)
            print(f"  Saved {output_path}: shape {combined.shape}")
