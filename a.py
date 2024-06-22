import pandas as pd
import numpy as np
df = pd.read_csv('/users/psen/preprocessed.csv', delimiter=',')
df_text = df['text']
print(type(df_text))
print(len(df_text))
print(type(np.asarray(df_text)))
