import pandas as pd
import numpy as np
#print(pd.__version__)

Car_Dataset = pd.read_csv("C:\\Users\\AnukyS\\Desktop\\car_fuel_efficiency.csv")


#2
#row_count = len(Car_Dataset)
#print(row_count)

#3
#print(Car_Dataset["fuel_type"].unique())

#4
#print(Car_Dataset.isnull().sum())


#5

# From_Asia = Car_Dataset[Car_Dataset["origin"] == "Asia"]
# max = From_Asia["fuel_efficiency_mpg"].max()
# print( max)

#6

#First_Horse_Power_Median = Car_Dataset["horsepower"].median()
#print(First_Horse_Power_Median)

# most_frequent = Car_Dataset["horsepower"].mode()
# #print(most_frequent)

# Car_Dataset["horsepower"] = Car_Dataset["horsepower"].fillna(most_frequent)

# Second_Horse_power_median = Car_Dataset["horsepower"].median()
# print(Second_Horse_power_median)


#7

Asia_Cars = Car_Dataset[Car_Dataset['origin'] == 'Asia']
#print(Asia_Cars)

new = Asia_Cars[['vehicle_weight', 'model_year']]
new7 = new.iloc[:7, :]
#print(new7)
X = new7.to_numpy()
XTX = X.T @ X
print(XTX)
XTX_inv = np.linalg.inv(XTX)
Y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200]).reshape(-1, 1)


w = XTX_inv @ X.T @ Y
sum_w = np.sum(w)
print(sum_w)