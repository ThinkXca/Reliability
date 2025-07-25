import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import numpy as np


class RULSurvivalDataset(Dataset):
    def __init__(self, df, feature_cols, scaler=None, is_train=True):
        self.samples = []
        grouped = df.groupby('unit_nr')

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(df[feature_cols])
        self.scaler = scaler

        for pid, group in grouped:
            group = group.sort_values('time_cycle')
            features = group[feature_cols].values
            features = self.scaler.transform(features)
            t_seq = group['time_cycle'].values
            t_seq = [t_seq[0]] + list(np.diff(t_seq))  # time intervals
            t_seq = [t * 1.0 for t in t_seq]  # optional scaling

            duration = group['max_time'].values[-1] - group['time_cycle'].values[-1]
            event = group['event'].values[-1]

            sample = {
                'pid': pid,
                't_seq': torch.tensor(t_seq, dtype=torch.float),
                'features': torch.tensor(features, dtype=torch.float),
                'duration': torch.tensor(duration, dtype=torch.float),
                'event': torch.tensor(event, dtype=torch.float)
            }
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (sample['t_seq'], len(sample['t_seq']), sample['features'],
                sample['duration'], sample['event'], sample['pid'])


def collate_fn(batch):
    t_seqs, lengths, features, durations, events, pids = zip(*batch)

    padded_t_seqs = pad_sequence(t_seqs, batch_first=True, padding_value=0.0)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    durations = torch.stack(durations)
    events = torch.stack(events)

    return padded_t_seqs, lengths, padded_features, durations, events, pids


def get_data_loaders(csv_path='../../data/NASA-Turbofan/TLSTM/FD_Data_sampled.csv', batch_size=16, split_ratio=0.8):
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith('setting_') or col.startswith('s_')]

    # Shuffle unit IDs and split
    unit_ids = df['unit_nr'].unique()
    np.random.seed(42)  
    np.random.shuffle(unit_ids)
    split_idx = int(len(unit_ids) * split_ratio)
    train_ids = unit_ids[:split_idx]
    val_ids = unit_ids[split_idx:]

    train_df = df[df['unit_nr'].isin(train_ids)].copy()
    val_df = df[df['unit_nr'].isin(val_ids)].copy()

    # Fit scaler on train
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_dataset = RULSurvivalDataset(train_df, feature_cols, scaler=scaler)
    val_dataset = RULSurvivalDataset(val_df, feature_cols, scaler=scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader
