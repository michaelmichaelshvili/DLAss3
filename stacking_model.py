import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import concatenate, Dense
from DataPreparation import load_product_title_search_term_lists
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


window_size=20
product_title_input, search_term_input = load_product_title_search_term_lists('train', window_size)
train_labels = pd.read_csv(r"C:\Users\micha\Desktop\train.csv", encoding="ISO-8859-1")['relevance']


siamise_model = load_model('model.h5')

siamise_model.layers.pop()
# siamise_model.layers.pop()
# x = siamise_model.output
# x = concatenate([x[0], x[1]])
# prediction = Dense(1, activation='sigmoid')(x)
predictions = siamise_model.predict([product_title_input,search_term_input])

print('Done predict')
x_train = predictions[:59000]
x_val = predictions[59000:]
y_train = train_labels[:59000]
y_val = train_labels[59000:]


rfr = RandomForestRegressor(n_estimators=15, max_depth=30, min_samples_split=25, random_state=123,n_jobs=-1)
rfr.fit(x_train, y_train)
val_pred = rfr.predict(x_train)
print(f'Validation score (rmse): {np.sqrt(mean_squared_error(val_pred,y_train))}')

val_pred = rfr.predict(x_val)
print(f'Validation score (rmse): {np.sqrt(mean_squared_error(val_pred,y_val))}')

print('Done rfr')

knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(x_train, y_train)
val_pred = knn.predict(x_train)
print(f'Validation score (rmse): {np.sqrt(mean_squared_error(val_pred,y_train))}')

val_pred = knn.predict(x_val)
print(f'Validation score (rmse): {np.sqrt(mean_squared_error(val_pred,y_val))}')


