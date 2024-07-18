# -*- coding: utf-8 -*-
"""ELENet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_rjvXlXacL8Dr7VvEQXLcSD3odwmldqq
"""

import numpy as np
import pandas as pd
import re

df = pd.read_csv("/content/modified_file 1.csv")

df.columns

drop_columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                '30', '31', '32', '33', '34', 'Ego_pathlane', 'A', 'C', 'index1', 'E', 'F', 'index2',
                'H', 'I', 'index3', 'K', 'L', 'index4', 'N', 'O', 'index5']
df = df.drop(columns=drop_columns)
print(df.columns)

df.dtypes



def extract_tuples(row):
    tuples = re.findall(r'\(([^,]+),([^\)]+)\)', row)
    return [(str(signature), float(distance)) for signature, distance in tuples]

df['tuples'] = df['detection_signature'].apply(extract_tuples)

def normalize_distances(tuples):
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


df['tuples_norm'] = df['tuples'].apply(normalize_distances)

def create_right_encoded_vectors(tuples_norm):
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
            encoded_vectors.append(encoded_vector)
        else:
            encoded_vectors.append([-float('inf')] * len(standard_format))

    return encoded_vectors

df['right_encoded'] = df['tuples_norm'].apply(create_right_encoded_vectors)

def create_left_encoded_vectors(tuples_norm):
    standard_format = ['RB', 'DW', 'DY', 'SW', 'SY']
    encoded_vectors = []

    signatures, distances = zip(*tuples_norm)
    c_index = signatures.index('C')

    for i in range(0, c_index):
        if i < len(signatures):
            signature = signatures[i]
            distance = distances[i]

            encoded_vector = [float('inf')] * len(standard_format)
            encoded_vector[standard_format.index(signature)] = distance
            encoded_vectors.append(encoded_vector)
        else:
            encoded_vectors.append([float('inf')] * len(standard_format))

    while len(encoded_vectors) < 8:
        encoded_vectors.append([float('inf')] * len(standard_format))

    return encoded_vectors

df['left_encoded'] = df['tuples_norm'].apply(create_left_encoded_vectors)

q=df['detection_signature'].iloc[5678]
q

a=df['tuples'].iloc[5678]
a

b=df['tuples_norm'].iloc[6678]
b

d=df['right_encoded'].iloc[6678]
d

e=df['left_encoded'].iloc[6678]
e

def lstm_data_transform(x_data, y_data, num_steps=Window_size):
    x_array = np.array([x_data[i:i + num_steps] for i in range(x_data.shape[0] - num_steps)])
    y_array = y_data[num_steps-1:]
    return x_array, y_array

import re
import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df

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
                encoded_vectors.append(encoded_vector)
            else:
                encoded_vectors.append([-float('inf')] * len(standard_format))

        return encoded_vectors

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

                encoded_vector = [float('inf')] * len(standard_format)
                encoded_vector[standard_format.index(signature)] = distance
                encoded_vectors.append(encoded_vector)
            else:
                encoded_vectors.append([float('inf')] * len(standard_format))

        while len(encoded_vectors) < 8:
            encoded_vectors.append([float('inf')] * len(standard_format))

        return encoded_vectors

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

# Create an instance of the class
data_processor = DataProcessor(df)
# Call the apply_extract_tuples method
data_processor.apply_extract_tuples(column_name='detection_signature')
# Call the apply_normalize_distances method
data_processor.apply_normalize_distances(column_name='tuples')
# Call the apply_create_right_encoded_vectors method
data_processor.apply_create_right_encoded_vectors(column_name='tuples_norm')
# Call the apply_create_left_encoded_vectors method
data_processor.apply_create_left_encoded_vectors(column_name='tuples_norm')

