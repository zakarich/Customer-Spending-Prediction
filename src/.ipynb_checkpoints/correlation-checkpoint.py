from sklearn.preprocessing import LabelEncoder

import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np


def data_encoding(df, categorical_feature):
    le = LabelEncoder()
    for col in categorical_feature:
        df[col] = le.fit_transform(df[col])
        
def corr_matrix(df): 
    df_corr = df.corr()
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = df_corr.columns,
            y = df_corr.index,
            z = np.array(df_corr),
            colorscale = 'teal',
            text=np.array(df_corr),
            texttemplate="%{text}"
        )
    )
    fig.show()