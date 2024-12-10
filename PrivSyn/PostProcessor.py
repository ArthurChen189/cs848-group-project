import yaml
import json
import copy
import numpy as np
import pandas as pd


class RecordPostprocessor:
    def __init__(self, data, configpath, datainfopath, decode_mapping):
        self.configpath = configpath
        with open(configpath, 'r', encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.datainfopath = datainfopath
        with open(datainfopath, 'r', encoding="utf-8") as f:
            datainfopath = json.load(f)
        self.datatypes = datainfopath['dtype']
        self.fillna = datainfopath['fillna']

        self.data = copy.deepcopy(data)
        self.decode_mapping = decode_mapping
        self.processed_data = data.copy(deep = True)

    def post_process(self):
        # Applying ungrouping to grouped attributes (Optional)
        # self.data = self.ungrouping_attributes(self.config['grouping_list'], self.data)
        # Applying unbinning to binned types
        self.data = self.unbinning_attributes(self.config['binning_list'], self.data)
        # Decoding the remaining columns
        self.data = self.decode_remaining(self.config, self.data)
        # Ensuring data types are of the correct type
        self.data = self.ensure_types(self.datatypes, self.data)
        return self.data

    def unbinning_attributes(self, binning_info, data):
        for att, spec_list in binning_info.items():
            [s, t, step] = spec_list
            bins = np.r_[-np.inf, np.arange(int(s), int(t), int(step)), np.inf]
            bins[0] = bins[1] - 1
            bins[-1] = bins[-2] + 2
            values_map = {i: int((bins[i] + bins[i + 1]) / 2) for i in range(len(bins) - 1)}
            data[att] = data[att].map(values_map)
        return data

    def ungrouping_attributes(self, grouping_list, data):
        for grouping in grouping_list :
            if grouping is None:
                return data
            grouped_attribute = grouping['grouped_name']
            attributes = grouping['attributes']
            mapping = pd.Index(self.decode_mapping[grouped_attribute])
            data[grouped_attribute] = pd.MultiIndex.to_numpy(mapping[data[grouped_attribute]])
            data[attributes] = pd.DataFrame(data[grouped_attribute].tolist(), index=data.index)
            data = data.drop(grouped_attribute, axis=1)
        return data

    def decode_remaining(self, config, data):
        decoded_attributes = list(config['binning_list'].keys())
        for attr, mapping in self.decode_mapping.items():
            if attr in decoded_attributes:
                continue
            try:
                # Convert mapping to a pandas Series for easier indexing
                mapping_series = pd.Series(mapping)
                
                if data[attr].isna().any():
                    # Create a copy to avoid modifying the original data
                    mapped_data = pd.Series(index=data.index, dtype=object)
                    # Map non-NaN values
                    non_nan_mask = ~data[attr].isna()
                    mapped_data[non_nan_mask] = mapping_series.iloc[data[attr][non_nan_mask].astype(int)].values
                    # Fill NaN values
                    mapped_data[~non_nan_mask] = self.fillna[attr]
                    data[attr] = mapped_data
                else:
                    # If no NaN values, proceed with simple mapping
                    data[attr] = mapping_series.iloc[data[attr].astype(int)].values
                    
            except Exception as e:
                print(f"\nError decoding column '{attr}':")
                print(f"Mapping values: {mapping[:5]}...")
                print(f"Data values: {data[attr].value_counts().head()}")
                print(f"Data dtype: {data[attr].dtype}")
                print(f"Fill NA value: {self.fillna[attr]}")
                raise Exception(f"Error decoding column '{attr}': {str(e)}")
        return data

    def ensure_types(self, datatypes, data):
        for col, data_type in datatypes.items():
            data[col] = data[col].astype(data_type)
        return data