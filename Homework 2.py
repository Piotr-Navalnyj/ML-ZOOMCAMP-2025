import pandas as pd
import numpy as np

CarData = pd.read_csv("C:\\Users\\AnukyS\\Desktop\\car_fuel_efficiency.csv")
# len(CarData)

#1

# print(CarData.isnull().sum())

print(CarData)

#2
# median_horsepower = CarData['horsepower'].median()
# print(median_horsepower)

#3
# Cut_Car = CarData[['engine_displacement',
# 'horsepower',
# 'vehicle_weight',
# 'model_year',
# 'fuel_efficiency_mpg']]
# #print(Cut_Car)

# shuffled_Cut_Car = Cut_Car.sample(frac=1, random_state=42).reset_index(drop=True)
# # #print(shuffled_Cut_Car)

# # #print(len(shuffled_Cut_Car))

# n = len(shuffled_Cut_Car)
# n_val = int(n * 0.2)
# n_test = int(n * 0.2)
# n_train = n - n_val - n_test
# # print(n_train, n_val, n_test)

# Car_Cut_train = shuffled_Cut_Car.iloc[:n_train].copy()
# Car_Cut_val = shuffled_Cut_Car.iloc[n_train:n_train+n_val].copy()
# # Cat_Cur_test = shuffled_Cut_Car.iloc[n_train+n_val:].copy()



# features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
# target = 'fuel_efficiency_mpg'


# X_train_0 = Car_Cut_train[features].copy()
# X_val_0 = Car_Cut_val[features].copy()

# X_train_0['horsepower'] = X_train_0['horsepower'].fillna(0)
# X_val_0['horsepower'] = X_val_0['horsepower'].fillna(0)

# y_train = Car_Cut_train[target].values
# y_val = Car_Cut_val[target].values


# X_train_0 = np.hstack([np.ones((len(X_train_0), 1)), X_train_0.values])
# X_val_0 = np.hstack([np.ones((len(X_val_0), 1)), X_val_0.values])


# w_0 = np.linalg.inv(X_train_0.T @ X_train_0) @ X_train_0.T @ y_train


# y_pred_0 = X_val_0 @ w_0
# rmse_0 = np.sqrt(np.mean((y_val - y_pred_0) ** 2))
# print("RMSE (fill 0):", round(rmse_0, 2))




# X_train_mean = Car_Cut_train[features].copy()
# X_val_mean = Car_Cut_val[features].copy()

# horsepower_mean = X_train_mean['horsepower'].mean()
# X_train_mean['horsepower'] = X_train_mean['horsepower'].fillna(horsepower_mean)
# X_val_mean['horsepower'] = X_val_mean['horsepower'].fillna(horsepower_mean)


# X_train_mean = np.hstack([np.ones((len(X_train_mean), 1)), X_train_mean.values])
# X_val_mean = np.hstack([np.ones((len(X_val_mean), 1)), X_val_mean.values])


# w_mean = np.linalg.inv(X_train_mean.T @ X_train_mean) @ X_train_mean.T @ y_train


# y_pred_mean = X_val_mean @ w_mean
# rmse_mean = np.sqrt(np.mean((y_val - y_pred_mean) ** 2))
# print("RMSE (fill mean):", round(rmse_mean, 2))

                                                                                                            #4




# X_train_0['horsepower'] = X_train_0['horsepower'].fillna(0)
# X_val_0['horsepower']   = X_val_0['horsepower'].fillna(0)

# y_train = Car_Cut_train[target].values
# y_val   = Car_Cut_val[target].values


# X_train_0 = np.hstack([np.ones((len(X_train_0), 1)), X_train_0.values])
# X_val_0   = np.hstack([np.ones((len(X_val_0), 1)), X_val_0.values])


# r_list = [0, 0.01, 0.1, 1, 5, 10, 100]


# rmse_results = []


# I = np.eye(X_train_0.shape[1])
# I[0, 0] = 0   


# for r in r_list:
    
#     w_r = np.linalg.inv(X_train_0.T @ X_train_0 + r * I) @ X_train_0.T @ y_train

   
#     y_pred = X_val_0 @ w_r
#     rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
#     rmse_results.append((r, round(rmse, 2)))


# for r, rmse in rmse_results:
#     print(f"r = {r:<6}  -->  RMSE = {rmse}")


                                                                #5



# Cut_Car = CarData[['engine_displacement',
#                    'horsepower',
#                    'vehicle_weight',
#                    'model_year',
#                    'fuel_efficiency_mpg']]


# features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
# target = 'fuel_efficiency_mpg'


# seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# rmse_scores = []

# for seed in seeds:
    
#     shuffled = Cut_Car.sample(frac=1, random_state=seed).reset_index(drop=True)
    
   
#     n = len(shuffled)
#     n_val = int(n * 0.2)
#     n_test = int(n * 0.2)
#     n_train = n - n_val - n_test


#     train = shuffled.iloc[:n_train].copy()
#     val   = shuffled.iloc[n_train:n_train + n_val].copy()
    
#     X_train = train[features].copy()
#     X_val   = val[features].copy()
#     y_train = train[target].values
#     y_val   = val[target].values

   
#     X_train['horsepower'] = X_train['horsepower'].fillna(0)
#     X_val['horsepower']   = X_val['horsepower'].fillna(0)

  
#     X_train = np.hstack([np.ones((len(X_train), 1)), X_train.values])
#     X_val   = np.hstack([np.ones((len(X_val), 1)), X_val.values])

  
#     w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    
#     y_pred = X_val @ w
#     rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
#     rmse_scores.append(rmse)

# std_rmse = round(np.std(rmse_scores), 3)

# print("RMSE scores for different seeds:", [round(r, 2) for r in rmse_scores])
# print(f"Standard deviation of RMSEs: {std_rmse}")


                                                                #6


Cut_Car = CarData[['engine_displacement',
                   'horsepower',
                   'vehicle_weight',
                   'model_year',
                   'fuel_efficiency_mpg']]


features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'


seed = 9
shuffled = Cut_Car.sample(frac=1, random_state=seed).reset_index(drop=True)


n = len(shuffled)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test


train = shuffled.iloc[:n_train].copy()
val   = shuffled.iloc[n_train:n_train + n_val].copy()
test  = shuffled.iloc[n_train + n_val:].copy()


train_full = pd.concat([train, val]).reset_index(drop=True)


X_train = train_full[features].copy()
X_test  = test[features].copy()
y_train = train_full[target].values
y_test  = test[target].values


X_train['horsepower'] = X_train['horsepower'].fillna(0)
X_test['horsepower']  = X_test['horsepower'].fillna(0)


X_train = np.hstack([np.ones((len(X_train), 1)), X_train.values])
X_test  = np.hstack([np.ones((len(X_test), 1)), X_test.values])


r = 0.001


I = np.eye(X_train.shape[1])
I[0, 0] = 0


w_r = np.linalg.inv(X_train.T @ X_train + r * I) @ X_train.T @ y_train


y_pred_test = X_test @ w_r


rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

print(f"✅ RMSE on test dataset (r = {r}): {round(rmse_test, 2)}")