import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from DataPreparation import prepere_X_data



train = prepere_X_data()
train_labels = pd.read_csv(r"C:\Users\micha\Desktop\train.csv", encoding="ISO-8859-1")['relevance']

X_train, x_val = train[:59000], train[59000:]

y_train, y_val = train_labels[:59000], train_labels[59000:]

rfr = RandomForestRegressor(n_estimators=15, max_depth=30, min_samples_split=25, random_state=123,n_jobs=-1)

val_pred = rfr.predict(X_train)
print(f'Validation score (rmse): {np.sqrt(mean_squared_error(val_pred,y_train))}')

val_pred = rfr.predict(x_val)
print(f'Validation score (rmse): {np.sqrt(mean_squared_error(val_pred,y_val))}')



