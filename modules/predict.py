# <YOUR_IMPORTS>
import os
import pandas as pd
import json
import dill
from datetime import datetime


def predict():
    # <YOUR_CODE>
    path = os.environ.get('PROJECT_PATH', '.')
    path_model = f'{path}/data/models'
    files = os.listdir(path_model)
    if files:
        files = [os.path.join(path_model, file) for file in files]
        files = [file for file in files if os.path.isfile(file)]
        filename = max(files, key=os.path.getctime)

    with open(filename, 'rb') as file:
        model = dill.load(file)

    pred_lst = []

    for test_file in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', test_file)) as f:
            data = json.load(f)
            df = pd.DataFrame.from_dict([data])
            y = model.predict(df)
            pred_lst.append((df['id'][0], y[0]))

    pred_dict = {x[0]: x[1] for x in pred_lst}
    result_df = pd.DataFrame.from_dict(pred_dict, orient='index').reset_index()
    result_df = result_df.rename(columns={'index': 'id', 0: 'price_category'})
    result_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
