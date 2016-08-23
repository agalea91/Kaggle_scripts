import numpy as np
import pandas as pd

types = {'Semana':np.uint8, #'Agencia_ID':np.uint16, 'Canal_ID':np.uint8, 'Ruta_SAK':np.uint16,
         'Cliente_ID':np.uint32, 'Producto_ID':np.uint16, 'Demanda_uni_equil':np.uint32}

df = pd.read_csv('../input/train.csv', usecols=types.keys(),
                 dtype=types)

weeks = np.unique(df.Semana.values)
week_labs = ['W'+str(i) for i in weeks]

# Initialize training dataframe
df_train = df[(df.Semana == weeks[0])]\
    .drop('Semana', axis=1)\
    .rename(columns={'Demanda_uni_equil': week_labs[0]})
# Cut down other df to save memory
df = df[(df.Semana != weeks[0])]

for w, w_label in zip(weeks[1:2], week_labs[1:2]):
    df_train = df_train.merge(
        df[(df.Semana == w)]\
            .drop('Semana', axis=1)\
            .rename(columns={'Demanda_uni_equil': w_label}),
        how='outer',
        on=['Cliente_ID', 'Producto_ID'])
    df = df[(df.Semana != w)]

df_train.to_csv('train_df.csv')
