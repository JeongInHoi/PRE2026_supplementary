import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py
"""            if (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4 in [3, 5, 7] and c5 in [3, 5, 7]) or (t > last_reward + 10000 and c1 == 4 and c2 == 5 and c3 == 6 and c4 == 5 and c5 == 5) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 3 and c5 == 70) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 30) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 50) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 70) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 90) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 110) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 130) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 150) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 300) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 5 and c5 == 600) or (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4  == 30 and c5 == 30):"""
class TripletRaster:
    def __init__(self,e_spikes, i_spikes, reward_times, stimulus_times, preproc_path = None, train = True ):
        
        self.train = train
        if preproc_path and os.path.exists(preproc_path):
            self._load_from_h5(preproc_path)
        else: 
            self.e_spikes = pd.read_csv(e_spikes)
            self.i_spikes = pd.read_csv(i_spikes)
            self.reward_times = pd.read_csv(reward_times, header = None)
            self.stimulus_times = pd.read_csv(stimulus_times, header = None)
            self.filtered_stimuli = self._filter_valid_stimuli()
            self.labels = [self._compute_label(stim) for stim in self.filtered_stimuli]
            self._spike_arrays = self._preprocess()
            if preproc_path:
                self._save_to_h5(preproc_path)

        
    def _filter_valid_stimuli(self):
        last_reward = self.reward_times.iloc[:,0].max() if len(self.reward_times) > 0 else 0
        valid_rows = []
        for row in self.stimulus_times.values:
            t, c1, c2, c3, c4, c5, c6 = row[:7]
            if (t > last_reward + 10000 and c1 == 0 and c2 == 1 and c3 == 2 and c4 ==5 and c5 ==5 and c6 in [5,10,15,20,25,30,35,40,45,50]):
                valid_rows.append(row)
        num_data = len(valid_rows)
        num_train = int(num_data*4/5) 
        if self.train == True:
            valid_rows = valid_rows[:num_train]
        else :
            valid_rows = valid_rows[num_train:]    
        return np.array(valid_rows)
        
    def _compute_label(self, stim):
        #all_label = [33, 35, 37, 53, 55, 57, 73, 75, 77, 456, 370, 530, 550, 570, 590, 5110, 5130, 5150, 5300, 5600, 3030]
        all_label = [555,5510,5515,5520,5525,5530,5535,5540,5545,5550]
        if stim[6] == 5:
            raw_label = 555
        if stim[6] == 10:
            raw_label = 5510
        if stim[6] == 15:
            raw_label = 5515
        if stim[6] == 20:
            raw_label = 5520
        if stim[6] == 25:
            raw_label = 5525
        if stim[6] == 30:
            raw_label = 5530
        if stim[6] == 35:
            raw_label = 5535
        if stim[6] == 40:
            raw_label = 5540
        if stim[6] == 45:
            raw_label = 5545
        if stim[6] == 50:
            raw_label = 5550

        
        """if stim[1] == 4 and stim[2] == 5 and stim[3] == 6:
            raw_label = 456
        elif stim[4] == 3 and stim[5] == 70:
            raw_label = 370
        elif stim[4] == 5 and stim[5] == 30:
            raw_label = 530
        elif stim[4] == 5 and stim[5] == 50:
            raw_label = 550
        elif stim[4] == 5 and stim[5] == 70:
            raw_label = 570
        elif stim[4] == 5 and stim[5] == 90:
            raw_label = 590
        elif stim[4] == 5 and stim[5] == 110:
            raw_label = 5110
        elif stim[4] == 5 and stim[5] == 130:
            raw_label = 5130
        elif stim[4] == 5 and stim[5] == 150:
            raw_label = 5150
        elif stim[4] == 5 and stim[5] == 300:
            raw_label = 5300
        elif stim[4] == 5 and stim[5] == 600:
            raw_label = 5600
        elif stim[4] == 30 and stim[5] == 30:
            raw_label = 3030 
        else:
            raw_label = int(stim[4]) * 10 + int(stim[5])"""
            
        return all_label.index(raw_label)
    
    def _preprocess(self):
        _spike_arrays = []
        spike_times = self.e_spikes.iloc[:, 0].values
        spike_ids = self.e_spikes.iloc[:, 1].values
        for i,stim in enumerate(tqdm(self.filtered_stimuli, desc="Preprocessing dataset")):
            stim_time = stim[0]
            t_start = stim_time - 10
            t_end = stim_time + 100 #default 30
            start_idx = np.searchsorted(spike_times, t_start, side='left')
            end_idx   = np.searchsorted(spike_times, t_end, side='right')
            
            
            selected_times = spike_times[start_idx:end_idx]
            selected_units = spike_ids[start_idx:end_idx]
            
            # structured array (SHD 형식)
            spike_array = np.zeros(len(selected_times), dtype=[('t', '<i8'), ('x', '<i8'), ('p', '<i8')])
            spike_array['t'] = selected_times-t_start #시간을 0부터 시작하도록 바꿈
            spike_array['x'] = selected_units
            spike_array['p'] = 1  # fixed polarity
            _spike_arrays.append(spike_array)
        return _spike_arrays

    def _save_to_h5(self, path):
        with h5py.File(path, 'w') as f:
            f.create_dataset("labels", data=np.array(self.labels))
            g = f.create_group("spikes")
            for i, arr in enumerate(self._spike_arrays):
                g.create_dataset(str(i), data=arr)
    
    def _load_from_h5(self, path):
        with h5py.File(path, 'r') as f:
            self.labels = f["labels"][:].tolist()
            self._spike_arrays = []
            for i in range(len(self.labels)):
                self._spike_arrays.append(f["spikes"][str(i)][:])
        self.filtered_stimuli = [None] * len(self.labels)  # dummy for __len__
    
        
    def __getitem__(self, idx):
        return self._spike_arrays[idx], self.labels[idx]
    @property
    def sensor_size(self):
        return (800, 1, 1)
    @property
    def ordering(self):
        return ('t', 'x', 'p')
    def __len__(self):
        return len(self.filtered_stimuli)
    @property
    def classes(self):
        return np.array([b'555', b'5510', b'5515', b'5520', b'5525', b'5530', b'5535', b'5540', b'5545', b'5550'],dtype='|S6')
