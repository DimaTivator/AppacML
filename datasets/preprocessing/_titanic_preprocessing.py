import pandas as pd
from sklearn.preprocessing import LabelEncoder


def titanic_preprocessing(df: pd.DataFrame):
    df = df.set_index('PassengerId')
    df = df.drop(columns=['Name', 'Ticket', 'Cabin'])
    df = df.dropna()
    le = LabelEncoder()
    df.loc[:, 'Sex'] = le.fit_transform(df['Sex'])
    df.loc[:, 'Embarked'] = le.fit_transform(df['Embarked'])
    return df
