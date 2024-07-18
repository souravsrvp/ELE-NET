import numpy as np
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class DataProcessor:
    def __init__(self, df):
        self.df = df
        print(self.df.head())

    def extract_tuples(self, row):
        tuples = re.findall(r'\(([^,]+),([^\)]+)\)', row)
        return [(str(signature), float(distance)) for signature, distance in tuples]

    def apply_extract_tuples(self, column_name):
        self.df['tuples'] = self.df[column_name].apply(self.extract_tuples)

    def normalize_distances(self, tuples):
        distances = [distance for signature, distance in tuples]
        
        min_distance = min(distances)
        max_distance = max(distances)

        if max_distance == min_distance:
            normalized_distances = [(signature, 0.0) for signature, distance in tuples]
        else:
            normalized_distances = [
                (signature, round((2 * (distance - min_distance) / (max_distance - min_distance)) - 1, 5))
                for signature, distance in tuples
            ]

        return normalized_distances

    def apply_normalize_distances(self, column_name):
        self.df['tuples_norm'] = self.df['tuples'].apply(self.normalize_distances)

    def create_right_encoded_vectors(self, tuples_norm):
        standard_format = ['RB', 'DW', 'DY', 'SW', 'SY']
        encoded_vectors = []
        signatures, distances = zip(*tuples_norm)
        c_index = signatures.index('C')

        for i in range(c_index + 1, c_index + 9):
            if i < len(signatures):
                signature = signatures[i]
                distance = distances[i]
                encoded_vector = [-float('inf')] * len(standard_format)
                encoded_vector[standard_format.index(signature)] = distance

                encoded_vectors+=encoded_vector
            else:
                encoded_vectors+=[-float('inf')] * len(standard_format)
        
        return np.array(encoded_vectors, dtype=np.double).reshape(40)

    def apply_create_right_encoded_vectors(self, column_name):
        self.df['right_encoded'] = self.df['tuples_norm'].apply(self.create_right_encoded_vectors)

    def create_left_encoded_vectors(self, tuples_norm):
        standard_format = ['RB', 'DW', 'DY', 'SW', 'SY']
        encoded_vectors = []

        signatures, distances = zip(*tuples_norm)
        c_index = signatures.index('C')

        for i in range(0, c_index):
            if i < len(signatures):
                signature = signatures[i]
                distance = distances[i]

                encoded_vector = [-float('inf')] * len(standard_format)
                encoded_vector[standard_format.index(signature)] = distance
                encoded_vectors += encoded_vector
            else:
                encoded_vectors += [-float('inf')] * len(standard_format)

        while len(encoded_vectors) < 40:
            encoded_vectors += [-float('inf')]
        
        return np.array(encoded_vectors, dtype=np.double).reshape(40)

    def apply_create_left_encoded_vectors(self, column_name):
        self.df['left_encoded'] = self.df['tuples_norm'].apply(self.create_left_encoded_vectors)

    def lstm_data_transform(self, x_data, y_data, num_steps=4):
        x_array = np.array([x_data[i:i + num_steps] for i in range(x_data.shape[0] - num_steps)])
        y_array = y_data[num_steps-1:]
        return x_array, y_array

    def apply_lstm_data_transform(self, x_column_name, y_column_name, num_steps=4):
        x_data = self.df[x_column_name].values
        y_data = self.df[y_column_name].values

        x_array, y_array = self.lstm_data_transform(x_data, y_data, num_steps=num_steps)
        return x_array, y_array

def create_lstm_dataset(x_array, y_array, lookback):
    X, y = [], []
    x_array = x_array.to_numpy()
    y_array = y_array.to_numpy()
    x_last = list(map(lambda x: x[-1], x_array))
    x_new = np.empty((x_array.shape[0], x_array.shape[1]-1+x_array[0, -1].shape[0]))
    
    x_new[:, 0] = x_array[:, 0]
    x_new[:, 1:] = x_last
    
    for i in range(len(x_array)-lookback):
        feature = x_new[i:i+lookback]
        target = y_array[i+lookback]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class ELE_Dataset(Dataset):
    def __init__(self, data_path, device, lr, Window_Size=50):
        self.Window_size = Window_Size
        self.device=device
        self.lr = lr
        self.df = pd.read_csv(data_path)
        #drop_columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
        #        '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
        #        '30', '31', '32', '33', '34', 'Ego_pathlane', 'A', 'C', 'index1', 'E', 'F', 'index2',
        #        'H', 'I', 'index3', 'K', 'L', 'index4', 'N', 'O', 'index5']
        
        self.df = self.df[['relative_distance', 'detection_signature', ' idxLeft', ' idxRight']]
        #self.df = self.df.drop(columns=drop_columns)
        #print(self.df.columns)
        
        self.df = self.df[self.df[' idxLeft']!=-1]
        self.df = self.df[self.df[' idxRight']!=-1]
        
        self.processor = DataProcessor(self.df)
        self.processor.apply_extract_tuples(column_name='detection_signature')
        
        self.processor.apply_normalize_distances(column_name='tuples')
        
        self.processor.apply_create_right_encoded_vectors(column_name='tuples_norm')
        
        self.processor.apply_create_left_encoded_vectors(column_name='tuples_norm')
        left_x_columns = ['relative_distance', 'left_encoded']
        
        right_x_columns = ['relative_distance','right_encoded']
        left_y_columns = [' idxLeft']
        right_y_columns = [' idxRight']
        self.left_x_data = self.processor.df[left_x_columns]
        self.right_x_data = self.processor.df[right_x_columns]
        self.left_y_data = self.processor.df[left_y_columns]
        self.right_y_data = self.processor.df[right_y_columns]
        self.left_x, self.left_y = create_lstm_dataset(self.left_x_data, self.left_y_data, self.Window_size)
        self.right_x, self.right_y = create_lstm_dataset(self.right_x_data, self.right_y_data, self.Window_size)

    def __len__(self):
        return len(self.left_x)
    
    def __getitem__(self, idx):
        match self.lr:
            case 'left':
                return self.left_x[idx].to(torch.float32).to(self.device), self.left_y[idx].squeeze(0).to(self.device)
            case 'right':
                return self.right_x[idx].to(torch.float32).to(self.device), self.right_y[idx].squeeze(0).to(self.device)
            case _:
                return 
        #return self.left_x[idx].to(torch.float32).to(self.device), self.left_y[idx].squeeze(0).to(self.device), self.right_x[idx].to(torch.float32).to(self.device), self.right_y[idx].squeeze(0).to(self.device)
        