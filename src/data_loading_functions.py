import json
from pathlib import Path

import numpy as np
import pandas as pd
import re

print("testing")

class HCRL_original:

    def parse_can_log(file_path):
        """Parse CAN log file and return a pandas DataFrame"""
        data = []

        with open(file_path, 'r') as f:
            for line in f:
                # Extract fields using regex
                match = re.search(r'Timestamp:\s([\d.]+)\s+ID:\s(\w+)\s+(\w+)\s+DLC:\s(\d+)\s+(.*)', line)
                if match:
                    timestamp, can_id, flags, dlc, hex_data = match.groups()
                    data.append({
                        'Timestamp': float(timestamp),
                        'ID': can_id,
                        'Flags': flags,
                        'DLC': int(dlc),
                        'Data': hex_data.strip()
                    })

        df = pd.DataFrame(data)
        df["Attack"] = "R"  # Default attack label for log files
        return df

    

    def parse_can_csv(file_path):
        df_can = pd.read_csv(file_path, skipinitialspace=True)

        if 'ID dlc' in df_can.columns:
            df_can[['ID', 'DLC']] = df_can['ID dlc'].astype(str).str.split(r'\s+', n=1, expand=True)
            df_can = df_can.drop(columns=['ID dlc'])
        elif set(['Timestamp','ID','DLC','data1','data2','data3','data4','data5','data6','data7','data8','attack/nonattack']).issubset(df_can.columns):
            df_can = df_can.rename(columns={'attack/nonattack': 'Attack'})
        else:
            names = ['Timestamp','ID','DLC','data1','data2','data3','data4','data5','data6','data7','data8','Attack']
            df_can = pd.read_csv(file_path, header=None, names=names, skipinitialspace=True)

        unnamed = [c for c in df_can.columns if isinstance(c, str) and c.startswith('Unnamed')]
        if 'Attack' not in df_can.columns and unnamed:
            df_can = df_can.rename(columns={unnamed[-1]: 'Attack'})

        data_cols = sorted(
            [c for c in df_can.columns if isinstance(c, str) and c.lower().startswith('data')],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )

        def build_data(row):
            try:
                dlc = int(row['DLC'])
            except Exception:
                dlc = 0

            values = []
            for col in data_cols[:min(dlc, len(data_cols))]:
                val = row.get(col)
                if pd.isna(val):
                    continue
                s = str(val).strip()
                if s.lower() in ['', 'nan']:
                    continue
                values.append(s)

            return ' '.join(values).strip()

        def infer_attack(row):
            attack = row.get('Attack')
            if isinstance(attack, str) and attack.strip():
                return attack

            try:
                dlc = int(row['DLC'])
            except Exception:
                dlc = 0

            for col in data_cols[min(dlc, len(data_cols)):]:
                val = row.get(col)
                if pd.isna(val):
                    continue
                s = str(val).strip()
                if s.lower() in ['', 'nan']:
                    continue
                if not re.fullmatch(r'[0-9A-Fa-f]{2}', s):
                    return s

            return pd.NA

        if data_cols:
            df_can['Data'] = df_can.apply(build_data, axis=1)
            if 'Attack' not in df_can.columns:
                df_can['Attack'] = pd.NA
            df_can['Attack'] = df_can.apply(infer_attack, axis=1).combine_first(df_can['Attack'])
        else:
            df_can['Data'] = df_can.get('Data', '').astype(str).str.strip()
            if 'Attack' not in df_can.columns:
                df_can['Attack'] = pd.NA

        df_can['Timestamp'] = pd.to_numeric(df_can['Timestamp'], errors='coerce')
        df_can['DLC'] = pd.to_numeric(df_can['DLC'], errors='coerce').astype('Int64')

        return df_can[['Timestamp','ID','DLC','Data','Attack']]


def _is_attack_label(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value) == 1
    normalized = str(value).strip().upper()
    return normalized in ('T', '1', 'ATTACK', 'TRUE')


class ROAD:
    def parse_can_log(path):
        """
        Parse CAN log lines like:
        (1290000000.012715) can0 230#FD000002D4000400
        Returns a DataFrame with columns: Timestamp (float), ID (hex str), Data (hex str), ID_int (int)
        """
        pattern = re.compile(r'^\((?P<ts>\d+\.\d+)\)\s+\S+\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]+)')
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                m = pattern.match(line.strip())
                if not m:
                    continue
                ts = float(m.group('ts'))
                id_hex = m.group('id').upper()
                data_hex = m.group('data').upper()

                if data_hex:
                    data_bytes = [data_hex[i:i+2] for i in range(0, len(data_hex), 2)]
                    data_formatted = ' '.join(data_bytes)
                else:
                    data_formatted = ''

                try:
                    id_int = int(id_hex, 16)
                except ValueError:
                    id_int = None
                rows.append((ts, id_hex, data_formatted, id_int))
        return pd.DataFrame(rows, columns=['Timestamp', 'ID', 'Data', 'ID_int'])

    @staticmethod
    def load_metadata(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _injection_intervals(entry):
        intervals = entry.get('injection_interval')
        if not intervals:
            return []

        if isinstance(intervals[0], (list, tuple)):
            return [interval for interval in intervals if len(interval) >= 2]

        if len(intervals) >= 2:
            return [intervals]

        return []

    @staticmethod
    def apply_attack_labels(df, capture_name, metadata):
        """
        Set binary Attack labels using ROAD capture metadata.

        0 = normal traffic, 1 = injected attack frames.
        Rows outside injection_interval are 0; rows inside are 1.
        Accelerator captures have no injection_interval and remain all 0.
        """
        labeled = df.copy()
        labeled['Attack'] = 0

        entry = metadata.get(capture_name)
        if not entry or labeled.empty:
            return labeled

        start_time = labeled['Timestamp'].iloc[0]
        for interval in ROAD._injection_intervals(entry):
            start = start_time + interval[0]
            end = start_time + interval[1]
            labeled.loc[labeled['Timestamp'].between(start, end), 'Attack'] = 1

        labeled['Attack'] = labeled['Attack'].astype(int)
        return labeled

    @staticmethod
    def load_captures_from_directory(directory):
        """
        Load every .log file in a ROAD directory into its own labeled dataframe.

        Returns a list of (log_path, dataframe) tuples.
        """
        directory = Path(directory)
        metadata_path = directory / 'capture_metadata.json'
        metadata = ROAD.load_metadata(metadata_path) if metadata_path.exists() else {}

        captures = []
        for log_path in sorted(directory.glob('*.log')):
            df = ROAD.parse_can_log(log_path)
            df = ROAD.apply_attack_labels(df, log_path.stem, metadata)
            captures.append((log_path, df))
        return captures

    @staticmethod
    def load_dataset(road_root):
        """
        Load ambient and attack ROAD captures as separate lists of dataframes.

        Returns:
            {
                'ambient_paths': [...],
                'ambient_dfs': [...],
                'attack_paths': [...],
                'attack_dfs': [...],
            }
        """
        road_root = Path(road_root)
        ambient = ROAD.load_captures_from_directory(road_root / 'ambient')
        attacks = ROAD.load_captures_from_directory(road_root / 'attacks')
        return {
            'ambient_paths': [path for path, _ in ambient],
            'ambient_dfs': [df for _, df in ambient],
            'attack_paths': [path for path, _ in attacks],
            'attack_dfs': [df for _, df in attacks],
        }


class CAN_MIRGU:
    def parse_can_log(filepath):
        """
        Parse CAN log file in format: (timestamp) can0 ID#DATA attack_label
        """
        data = []

        def format_data_hex(payload):
            cleaned = re.sub(r'\s+', '', payload or '')
            if not cleaned:
                return ''
            if len(cleaned) % 2 != 0:
                cleaned += '0'
            return ' '.join(cleaned[i:i+2].lower() for i in range(0, len(cleaned), 2))

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Pattern: (timestamp) can0 ID#DATA attack_label
                match = re.match(r'\(([^)]+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f\s]*)\s+(\S+)', line)
                if match:
                    timestamp, can_id, data_hex, attack_label = match.groups()
                    data.append({
                        'Timestamp': float(timestamp),
                        'ID': can_id,
                        'Data': format_data_hex(data_hex),
                        'Attack': attack_label
                    })

        df = pd.DataFrame(data)
        df["Attack"] = df["Attack"].apply(
            lambda x: 'R' if x == '0' else 'T' if x == '1' else x if x in ['T', 'R'] else pd.NA
        )
        return df
      

class HCRL_survival_analysis:
    def parse_can_csv(file_path):
        return HCRL_original.parse_can_csv(file_path)


def sliding_windows_id_data(df, window_size=32, step=2, attack_label='T'):
    ids = df['ID'].to_numpy(dtype=object)
    data = df['Data'].to_numpy(dtype=object)

    if 'Attack' in df.columns:
        attack = df['Attack'].to_numpy(dtype=object)
        def label_window(window_attack_values):
            return attack_label if any(_is_attack_label(v) for v in window_attack_values) else 'Real'
    else:
        attack = None
        def label_window(window_attack_values):
            return attack_label

    n = len(df)
    windows = []
    window_labels = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        windows.append(np.column_stack((ids[start:end], data[start:end])))
        window_labels.append(label_window(attack[start:end] if attack is not None else []))

    return windows, np.array(window_labels, dtype=object)

from functools import reduce
from typing import Iterable, Optional

def merge_dataframes(dfs, method='concat', on=None, how='outer', axis=0,
                     ignore_index=True, drop_duplicates=False, dedup_subset=None):
    """
    Merge a list of pandas DataFrame objects.

    Parameters:
    - dfs: iterable of DataFrame
    - method: 'concat' (default) or 'merge' (iterative pairwise merge)
    - on: column name or list of column names to merge on (only for method='merge').
          If None for 'merge', the intersection of columns across all frames is used.
    - how: merge how for method='merge' ('left','right','inner','outer') or concat join behavior for pandas.concat
    - axis: axis for concat (0 rows, 1 columns)
    - ignore_index: passed to pd.concat or used to reset index after merge
    - drop_duplicates: if True, drop duplicate rows after merge/concat
    - dedup_subset: subset of columns to consider when dropping duplicates

    Returns:
    - merged DataFrame
    """
    dfs = list(dfs)
    if not dfs:
        return pd.DataFrame()

    if method == 'concat':
        merged = pd.concat(dfs, axis=axis, ignore_index=ignore_index, sort=False)
    elif method == 'merge':
        if len(dfs) == 1:
            merged = dfs[0].copy()
        else:
            if on is None:
                common = set(dfs[0].columns)
                for df in dfs[1:]:
                    common &= set(df.columns)
                if not common:
                    raise ValueError("No common columns found to merge on; specify 'on' explicitly.")
                on_cols = list(common)
            else:
                on_cols = on
            merged = dfs[0].copy()
            for df in dfs[1:]:
                merged = merged.merge(df, how=how, on=on_cols)
        if ignore_index:
            merged = merged.reset_index(drop=True)
    else:
        raise ValueError("method must be 'concat' or 'merge'")

    if drop_duplicates:
        merged = merged.drop_duplicates(subset=dedup_subset)

    return merged