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


import numpy as np

def sliding_windows_id_data(df, window_size=32, step=2, attack_label='T'):
    ids = df['ID'].to_numpy(dtype=object)
    data = df['Data'].to_numpy(dtype=object)
    attack = df['Attack'].to_numpy(dtype=object)

    n = len(df)
    num_windows = (n - window_size) // step + 1
    windows = []
    window_labels = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        windows.append(np.column_stack((ids[start:end], data[start:end])))
        window_labels.append(attack_label if 'T' in attack[start:end] else 'Real')

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