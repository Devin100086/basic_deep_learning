import pandas as pd
def read_data():
    train_data=pd.read_csv('./data/train.csv')
    del train_data['Id']
    train_data=train_data.fillna(train_data.mean())
    df=pd.get_dummies(train_data,dummy_na=True)
    return df
if __name__ == '__main__':
    read_data()