import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api  as sm
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.tsa.holtwinters as ets
from scipy.stats import chi2
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
import numpy.linalg as LA
import seaborn as sns
import scipy as sp
from sklearn.model_selection import train_test_split
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import signal
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
import time
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import numpy.linalg as LA
import seaborn as sns
import scipy as sp
from sklearn.model_selection import train_test_split
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import signal
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import STL
import time
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
from statsmodels.tsa.arima.model import ARIMA
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# load the dataset title humidity into df_huidity and remove first row as it contains null value
df_humidity1=pd.read_csv("humidity.csv",header=0)
df_humidity = df_humidity1[['datetime','Vancouver']]
df_humidity['datetime'] = pd.to_datetime(df_humidity['datetime'])
df_humidity = df_humidity.iloc[1:]
df_humidity = df_humidity.resample('H', on='datetime').mean()
print(df_humidity.head())
plt.figure(figsize=(15,8))
plt.plot(df_humidity['Vancouver'])
plt.title('Humidity in Vancouver')
plt.ylabel('Humidity in %')
plt.xlabel('Date')
plt.show()

# load the dataset title pressure into df_pressure and remove first row as it contains null value
df_pressure1=pd.read_csv("pressure.csv",header=0)
df_pressure = df_pressure1[['datetime','Vancouver']]
df_pressure['datetime'] = pd.to_datetime(df_pressure['datetime'])
df_pressure = df_pressure.iloc[1:]
df_pressure = df_pressure.resample('H', on='datetime').mean()
print(df_pressure.head())
plt.figure(figsize=(15,8))
plt.plot(df_pressure['Vancouver'])
plt.title('Pressure in Vancouver')
plt.ylabel('Pressure in hPa')
plt.xlabel('Date')
plt.show()

# load the datastet title wind_speed into df_wind_speed and remove first row as it contains null value
df_wind_speed1=pd.read_csv("wind_speed.csv",header=0)
df_wind_speed = df_wind_speed1[['datetime','Vancouver']]
df_wind_speed['datetime'] = pd.to_datetime(df_wind_speed['datetime'])
df_wind_speed = df_wind_speed.iloc[1:]
df_wind_speed = df_wind_speed.resample('H', on='datetime').mean()
print(df_wind_speed.head())
plt.figure(figsize=(15,8))
plt.plot(df_wind_speed['Vancouver'])
plt.title('Wind Speed in Vancouver')
plt.ylabel('Wind Speed in m/s')
plt.xlabel('Date')
plt.show()

# load the dataset title temperature into df_temperature and remove first row as it contains null value
df_temperature1=pd.read_csv("temperature.csv",header=0)
df_temperature = df_temperature1[['datetime','Vancouver']]
df_temperature['datetime'] = pd.to_datetime(df_temperature['datetime'])
df_temperature = df_temperature.iloc[1:]
df_temperature = df_temperature.resample('H', on='datetime').mean()
print(df_temperature.head())
plt.figure(figsize=(15,8))
plt.plot(df_temperature['Vancouver'])
plt.title('Temperature in Vancouver')
plt.ylabel('Temperature in Kelvin')
plt.xlabel('Date')
plt.show()


# load the dataset title wind_direction into df_wind_direction and remove first row as it contains null value
df_wind_direction1=pd.read_csv("wind_direction.csv",header=0)
df_wind_direction = df_wind_direction1[['datetime','Vancouver']]
df_wind_direction['datetime'] = pd.to_datetime(df_wind_direction['datetime'])
df_wind_direction = df_wind_direction.iloc[1:]
# Use Label Encoder to convert the wind direction into numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_wind_direction['Vancouver'] = le.fit_transform(df_wind_direction['Vancouver'])
df_wind_direction = df_wind_direction.resample('H', on='datetime').mean()
print(df_wind_direction.head())
plt.figure(figsize=(15,8))
plt.plot(df_wind_direction['Vancouver'])
plt.title('Wind Direction in Vancouver')
plt.ylabel('Wind Direction in degrees')
plt.xlabel('Date')
plt.show()

# load the dataset tite weather_description into df_weather_description and remove first row as it contains null value
df_weather_description1=pd.read_csv("weather_description.csv",header=0)
df_weather_description = df_weather_description1[['datetime','Vancouver']]
df_weather_description['datetime'] = pd.to_datetime(df_weather_description['datetime'])
df_weather_description = df_weather_description.iloc[1:]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_weather_description['Vancouver'] = le.fit_transform(df_weather_description['Vancouver'])
df_weather_description = df_weather_description.resample('H', on='datetime').mean()
print(df_weather_description.head())
plt.figure(figsize=(15,8))
plt.plot(df_weather_description['Vancouver'])
plt.title('Weather Description in Vancouver')
plt.ylabel('Weather Description')
plt.xlabel('Date')
plt.show()


# removing null values:
df_humidity['Vancouver'] = df_humidity['Vancouver'].interpolate(method='linear', limit_direction='forward', axis=0)
df_pressure['Vancouver'] = df_pressure['Vancouver'].interpolate(method='linear', limit_direction='forward', axis=0)
df_temperature['Vancouver'] = df_temperature['Vancouver'].interpolate(method='linear', limit_direction='forward', axis=0)
df_wind_speed['Vancouver'] = df_wind_speed['Vancouver'].interpolate(method='linear', limit_direction='forward', axis=0)
df_wind_direction['Vancouver'] = df_wind_direction['Vancouver'].interpolate(method='linear', limit_direction='forward', axis=0)
df_weather_description['Vancouver'] = df_weather_description['Vancouver'].interpolate(method='linear', limit_direction='forward', axis=0)


# Combine all the datasets into one dataset df_weather and make sure there is only once column datetime
df_weather = pd.merge(df_humidity, df_pressure, on='datetime')
df_weather = pd.merge(df_weather, df_temperature, on='datetime')
df_weather = pd.merge(df_weather, df_wind_speed, on='datetime')
df_weather = pd.merge(df_weather, df_wind_direction, on='datetime')
df_weather = pd.merge(df_weather, df_weather_description, on='datetime')
df_weather.columns = ['humidity','pressure','temperature','wind_speed','wind_direction','weather_description']
print(df_weather.head())

df_weather['pressure'] = df_weather['pressure'].fillna(df_weather['pressure'].mean())
# Check for null values in the dataset
print(df_weather.isnull().sum())

#
print(df_weather.head())
print(df_weather.shape)
print(df_weather.dtypes)
print(df_weather.describe())
print(df_weather.info())


# ACF and PACF plots:
def ACF_PACF_Plot(series, lags, series_name=None):
    fig, axs = plt.subplots(2, 1)
    if series_name is not None:
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=axs[0], title=f'ACF Plot of {series_name}')
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axs[1], title=f'PACF Plot of {series_name}')
    else:
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=axs[0])
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axs[1])
    fig.tight_layout()
    fig.show()

ACF_PACF_Plot(df_weather['temperature'], 50, 'temperature')

# correlation martrix of all the features against temperature feature
corr = df_weather.corr()
print(corr['temperature'].sort_values(ascending=False))

# create a heatmap of correlation matrix
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# plot the temperature data for Vancouver city for the entire duration of the dataset to get an idea of the data distribution and trend
plt.figure(figsize=(15,8))
plt.plot(df_weather['temperature'])
plt.title('Temperature in Vancouver')
plt.ylabel('Temperature in Kelvin')
plt.xlabel('Date')
plt.show()

# split the dataset into train and test
from sklearn.model_selection import train_test_split
X = df_weather[['humidity','pressure','wind_speed','wind_direction','weather_description']]
y = df_weather['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)




# Save df3 as final_df
final_df = df_weather["temperature"]
# convert final_df into dataframe and name the column as Vancouver
final_df = pd.DataFrame(final_df)
final_df.columns = ['Vancouver']
print(final_df.head())
print(final_df.shape)
print(final_df.isnull().sum())


#############################################################################################################
# Critical Functions
#############################################################################################################

# Stationary test and plot
def ADF_Cal(x):
    result = adfuller(x)
    print('*' * 50)
    print(f"The results of ADF test:")
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    print('*' * 50)

def kpss_test(timeseries):
    print('*' * 50)
    print(f'The results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    print("dusre part ka code")
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
    print('*' * 50)

def Cal_rolling_mean_var(data, column_name=None, isdataframe=True):
    roll_mean = []
    roll_var = []
    if isdataframe is True:
        for i in range(1, len(data)):
            rmean = pd.array([data[column_name].head(i).mean()])
            roll_mean = np.append(roll_mean, rmean)
            rvar = pd.array([data[column_name].head(i).var()])  # divided by n instead of n-1?
            # rvar = np.var((df[column_name].head(i)))
            roll_var = np.append(roll_var, rvar)
    else:
        for j in range(len(data)):
            roll_mean.append(np.mean(data[:j + 1]))
            roll_var.append(np.var(data[:j + 1]))

    fig, axs = plt.subplots(2)
    # dusre part ka
    axs[0].plot(roll_mean, label='Varying Mean')
    if column_name is None:
        axs[0].set_title(f'Rolling Mean')
    else:
        axs[0].set_title(f'Rolling Mean - {column_name}')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Magnitude')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(roll_var, label='Varying Variance')

    if column_name is None:
        axs[1].set_title(f'Rolling Mean')
    else:
        axs[1].set_title(f'Rolling Variance - {column_name}')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.show()

ADF_Cal(final_df['Vancouver'])
kpss_test(final_df['Vancouver'])
Cal_rolling_mean_var(final_df, column_name='Vancouver', isdataframe=True)
print("#"*50)
print("*"*50)
print("The data is not stationary")
print("*"*50)
print("#"*50)



# Strength of trend and seasonality
from statsmodels.tsa.seasonal import STL
def stl_decompese(Temp):
    stl = STL(Temp)
    res = stl.fit()

    T = res.trend
    S = res.seasonal
    R = res.resid
    O = res.observed

    adj_seasonal = Temp.dropna() - S

    plt.figure()
    plt.plot(Temp.dropna(), label='Original Data')
    plt.plot(adj_seasonal, label='Adjusted Seasonal')
    plt.title(f'Adjusted Data vs Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    detrended = Temp.dropna() - T

    plt.figure()
    plt.plot(Temp.dropna(), label='Original Data')
    plt.plot(detrended, label='Detrended Data')
    plt.title(f'Detrended Data vs Original Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    a = max(0, 1 - (np.var(R) / np.var(T + R)))
    print("The strength of trend for this data set is ", round(a, 3))

    b = max(0, 1 - (np.var(R) / np.var(S + R)))
    print("The strength of seasonality for this data set is ",round(b,3))
    return res

stl_decompese(final_df['Vancouver'])

# Solving the problem of non-stationarity



# ACF and PACF plots
def ACF_PACF_Plot(series, lags, series_name=None):
    fig, axs = plt.subplots(2, 1)
    if series_name is not None:
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=axs[0], title=f'ACF Plot of {series_name}')
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axs[1], title=f'PACF Plot of {series_name}')
    else:
        sm.graphics.tsa.plot_acf(series, lags=lags, ax=axs[0])
        sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axs[1])
    fig.tight_layout()
    fig.show()


#############################################################################################################
#
#############################################################################################################



#############################################################################################################
# ACF_PACF_Plot(new_df['Vancouver_1'], lags=50, series_name='Vancouver_1')

# Plot new_df and final_df together to see the difference. Plot them with different colours, lables and legends.
# plt.figure(figsize=(15, 5))
# plt.plot(final_df['Vancouver'], label='Original')
# plt.plot(new_df['Vancouver_1'], label='Differenced')
# plt.legend(loc='best')
# plt.title('Differenced Data')
# plt.show()



#############################################################################################################
# Differencing(Seasonal and non-seasonal)
#############################################################################################################
def seasonal_diff(y, order):
    m = order
    diff = []
    for i in range(0, order):
        diff.append(0)
    for t in range(m, len(y)):
        diff.append(y[t] - y[t-m])
    return diff

def differencing(series, season=1, order=1):
    order = int(order)
    if order > 1:
        temp = differencing(series, season, 1)
        return differencing(temp, season, order-1)
    elif order == 0:
        return series
    else:
        res = []
        for i in range(season, len(series)):
            res.append(series[i] - series[i-season])
        return np.array(res)

new_df_ss = seasonal_diff(final_df['Vancouver'].values, order=24)
new_df_ss = pd.DataFrame(new_df_ss, columns=['Vancouver_2'])
new_df_ss.index = pd.date_range(start='2012-10-02', periods=len(new_df_ss), freq='H')
ADF_Cal(new_df_ss['Vancouver_2'])
kpss_test(new_df_ss['Vancouver_2'])
Cal_rolling_mean_var(new_df_ss, column_name='Vancouver_2', isdataframe=True)
ACF_PACF_Plot(new_df_ss['Vancouver_2'], lags=50, series_name='Vancouver_2')
stl_decompese(new_df_ss['Vancouver_2'])
plt.figure(figsize=(15, 5))
plt.plot(final_df['Vancouver'], label='Original')
plt.plot(new_df_ss['Vancouver_2'], label='Seasonal Differenced')
plt.legend(loc='best')
plt.title(' Seasonal Differenced Data')
plt.show()
new_df_ss_1 = differencing(new_df_ss['Vancouver_2'], season=1, order=1)
new_df_ss_1 = pd.DataFrame(new_df_ss_1, columns=['Vancouver_2'])
new_df_ss_1.index = pd.date_range(start='2012-10-02', periods=len(new_df_ss_1), freq='H')
ADF_Cal(new_df_ss_1['Vancouver_2'])
kpss_test(new_df_ss_1['Vancouver_2'])
# Cal_rolling_mean_var(new_df_ss_1, column_name='Vancouver_2', isdataframe=True)
ACF_PACF_Plot(new_df_ss_1['Vancouver_2'], lags=50, series_name='Vancouver_2')
stl_decompese(new_df_ss_1['Vancouver_2'])
plt.figure(figsize=(15, 5))
plt.plot(final_df['Vancouver'], label='Original')
plt.plot(new_df_ss_1['Vancouver_2'], label='1st Differenced after Seasonal Differencing')
plt.legend(loc='best')
plt.title(' Seasonal Differenced Data')
plt.show()
# new_df_ss_1['Vancouver_2']

# # 2nd order differencing
# new_df2 = differencing(new_df['Vancouver_1'], season=1, order=1)
# new_df2 = pd.DataFrame(new_df2, columns=['Vancouver_2'])
# ADF_Cal(new_df2['Vancouver_2'])
# kpss_test(new_df2['Vancouver_2'])
# Cal_rolling_mean_var(new_df2, column_name='Vancouver_2', isdataframe=True)
#
# #3rd order differencing
# new_df3 = differencing(new_df2['Vancouver_2'], season=1, order=1)
# new_df3 = pd.DataFrame(new_df3, columns=['Vancouver_3'])
# ADF_Cal(new_df3['Vancouver_3'])
# kpss_test(new_df3['Vancouver_3'])
# Cal_rolling_mean_var(new_df3, column_name='Vancouver_3', isdataframe=True)


#############################################################################################################
# Feature Selectioin: PCA, OLS, Standardization
#############################################################################################################



X = df_weather[['humidity','pressure','wind_speed','wind_direction','weather_description']]
y = df_weather['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#############################################################################################################
# Standardization
#############################################################################################################


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#############################################################################################################
# PCA
#############################################################################################################


pca = PCA(n_components="mle")
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
# print(f'Explained variance ratio: {explained_variance}')
# Print results of PCA and explained variance
print(f'Number of components: {pca.n_components_}')
print(f'PCA components: {pca.components_}')
print(f'PCA explained variance: {pca.explained_variance_}')
print(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')
print(f'PCA singular values: {pca.singular_values_}')

# Plot PCA
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()




#############################################################################################################
# Singular Value Decomposition
#############################################################################################################


s,d,v = np.linalg.svd (X_train, full_matrices = True)
print(f'singular values of x are {d}')
print(f'The condition number for x is {LA.cond(X_train)}')

#############################################################################################################
# Multiple Linear Regression(OLS)
#############################################################################################################


X_train = sm.add_constant(X_train, prepend=False)
model = sm.OLS(y_train, X_train).fit()
print(model.summary())
print("#"*50)
print("*"*50)
print("t-test p-values for all features: \n", model.pvalues)
print("F-test for final model: \n", model.f_pvalue)
print("R-squared for final model: \n", model.rsquared)
print("R-squared adjusted for final model: \n", model.rsquared_adj)
print("AIC for final model: \n", model.aic)
print("BIC for final model: \n", model.bic)
print("*"*50)
print("#"*50)

#Predictions
prediction = model.predict(X_train)


# HOLT WINTERS METHOD
X = df_weather[['humidity','pressure','wind_speed','wind_direction','weather_description']]
y = df_weather['temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


#############################################################################################################
# Critical Fucntions for performance metrics, plottig and forecasting
# Holt Winters forecasting function
#############################################################################################################



def autocorrelation(lags,list):
    val = [1]
    lag = [0]
    den = 0
    num = 0
    for k in range (0,len(list)):
        d = (list[k] - list.mean())**2
        den+= d
    for i in range(1,lags+1):
        num = 0
        for j in range (i,len(list)):
            n = ((list[j] - list.mean()) * (list[abs(j-i)]- list.mean()))
            num+= n
        cor = num/den
        val+=[cor]
        lag += [i]
    lag = np.array(lag)
    val = np.array(val)
    vals = val
    vals = vals[vals != 1]
    val1 = val[1:]
    lag1 = np.negative(lag[1:])
    val = np.concatenate((val1[::-1],val), axis=None)
    lag = np.concatenate((lag1[::-1], lag), axis=None)
    val = val[val != -1]
    return val,lag,vals

def holt_winters_method(train,test):
    holtt1 = ets.ExponentialSmoothing(train, trend='mul', damped_trend=False, seasonal='mul', seasonal_periods=30).fit()
    pred_train_holts = holtt1.predict(start=0, end=(len(train) - 1))
    pred_test_holts = holtt1.forecast(steps=len(test))

    residual_err = train - pred_train_holts
    forecast_err = test - pred_test_holts

    return pred_train_holts, pred_test_holts, residual_err, forecast_err
# #
# def plot_model_results(train, test,residual, forecast, label):
#     fig, ax = plt.subplots()
#     ax.plot(train, label='Train')
#     ax.plot(test, label='Test')
#     ax.plot(residual, label=f'{label} Method 1 step prediction')
#     ax.plot(forecast, label=f'{label} Method h-step prediction')
#     plt.legend()
#     plt.grid()
#     plt.title(f'{label} Method & Forecast of Temperature')
#     plt.xlabel('Time')
#     plt.ylabel('Values')
#     plt.show()

def plot_model_results(train,test,pred_test,pred_train,title):
    plt.figure(figsize=(10, 10))
    # plt.plot(train, label='training set', markerfacecolor='blue')
    # plt.plot([None for i in train] + [x for x in test], label='test set')
    # plt.plot([None for i in train] + [x for x in pred_test], label='h-step forecast')
    # train.plot(label='training set')
    # test.plot(label='test set')
    # pred_train.plot(label='1-step forecast')
    # pred_test.plot(label='h-step forecast')
    fig, ax = plt.subplots()
    ax.plot(train, label='Train')
    ax.plot(pred_train, label=f'{title} Method 1 step prediction')
    ax.plot(test, label='Test')
    ax.plot(pred_test, label=f'{title} Method h-step prediction')
    plt.legend()
    plt.title('Temperature by ' + title)
    plt.ylabel('Values')
    plt.xlabel('Time')
    plt.grid()
    plt.show()



def calculate_tests(residual_err, forecast_err,df_train,  title):
    """
    Calculates the F-test, t-test, AIC, BIC, RMSE, R-squared, adjusted R-squared, chi-square test, and MSE.

    Args:
        residual_err (numpy.ndarray): The residual errors.
        forecast_err (numpy.ndarray): The forecast errors.
        title (str): The name of the model.

    Returns:
        dict: A dictionary containing the results of the tests.

    """
    # print the shape of the residual errors, forecast errors, and training data.
    print(f"Residual errors shape: {residual_err.shape}")
    print(f"Forecast errors shape: {forecast_err.shape}")
    print(f"Training data shape: {df_train.shape}")
    # Calculate the F-test.
    F_statistic, p_value1 = stats.f_oneway(residual_err, forecast_err)

    # Calculate the t-test.
    # t_statistic, p_value2 = stats.ttest_rel(residual_err, df_train)

    # Calculate the AIC.
    aic = len(residual_err) * np.log(np.var(residual_err)) + 2 * len(residual_err)

    # Calculate the BIC.
    bic = len(residual_err) * np.log(np.var(residual_err)) + len(residual_err) * np.log(len(residual_err))

    # Calculate the RMSE.
    rmse = np.sqrt(np.mean(residual_err ** 2))

    # Calculate the R-squared.
    r_squared = 1 - np.var(residual_err) / np.var(df_train[2:])

    # Calculate the adjusted R-squared.
    adjusted_r_squared = 1 - (1 - r_squared) * (len(df_train) - 1) / (len(df_train) - len(df_train[2:]))

    # Calculate the MSE.
    mse = np.mean(residual_err ** 2)

    # Calculate the chi-square test.
    chi_statistic, p_value = stats.chisquare(residual_err)

    # Q Value
    Q = sm.stats.acorr_ljungbox(residual_err, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]

    # Print the results.
    print(f'F-statistic for {title}: {F_statistic}')
    print(f'p-value of F test for {title}: {p_value1}')
    # print(f't-statistic for {title}: {t_statistic}')
    # print(f'p-value of T Test for {title}: {p_value2}')
    print(f'AIC for {title}: {aic}')
    print(f'BIC for {title}: {bic}')
    print(f'RMSE for {title}: {rmse}')
    print(f'R-squared for {title}: {r_squared}')
    print(f'Adjusted R-squared for {title}: {adjusted_r_squared}')
    print(f'MSE for {title}: {mse}')
    print(f'Chi-square statistic for {title}: {chi_statistic}')
    print(f'Q-value for {title}: {Q}')

    return {
        'F_statistic': F_statistic,
        'p_value': p_value,
        'AIC': aic,
        'BIC': bic,
        'RMSE': rmse,
        'R_squared': r_squared,
        'adjusted_R_squared': adjusted_r_squared,
        'MSE': mse,
        'chi_statistic': chi_statistic,
        'p_value': p_value,
    }


def plot_graph(residual_err, title):
    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by ' + title)
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

# df_train_temp, df_test_temp = train_test_split(df_weather['temperature'], test_size=0.2)

#############################################################################################################
# Holt-Winters Method
#############################################################################################################



split_index = int(len(df_weather) * 0.8)  # Index to split at (80% of the data)
df_train_temp = df_weather['temperature'][:split_index]  # First 80% as training
df_test_temp = df_weather['temperature'][split_index:]  # Last 20% as testing

pred_train, pred_test, residual_error, forecast_error = holt_winters_method(df_train_temp.values, df_test_temp.values)
pred_train = pd.DataFrame(pred_train, index=df_train_temp.index)
pred_train = pred_train.sort_index()
pred_test = pd.DataFrame(pred_test, index=df_test_temp.index)
pred_test = pred_test.sort_index()
# residual_error = pd.DataFrame(residual_error, index=df_train_temp.index)
# forecast_error = pd.DataFrame(forecast_error, index=df_test_temp.index)

plot_model_results(df_train_temp, df_test_temp, pred_test, pred_train, 'Holt-Winter Method')
plot_graph(residual_error, 'Holt-Winter Method')
holt1 = calculate_tests(residual_error, forecast_error, df_train_temp, 'Holt-Winter Method')

df_train_temp1, df_test_temp1 = train_test_split(df_weather['temperature'], test_size=0.2)
pred_train1, pred_test1, residual_error1, forecast_error1 = holt_winters_method(df_train_temp1.values, df_test_temp1.values)
plot_graph(residual_error1, 'Holt-Winter Method')


#############################################################################################################
# Base Models Functions
#############################################################################################################



def average_forecast_method(arr_train, arr_test):
    pred_train = []
    pred_test = []

    arr_train = np.array(arr_train)
    arr_test = np.array(arr_test)

    for i in range(1, len(arr_train) + 1):
        avg = arr_train[:(i)].sum() / (len(arr_train[:(i)]))
        pred_train.append(round(avg, 2))

    for j in range(len(arr_test)):
        pred_test.append(pred_train[-1])

    pred_train.pop()

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)
    residual_err = arr_train[1:] - pred_train
    forecast_err = arr_test - pred_test

    return pred_train, pred_test, residual_err, forecast_err


def naive_method(arr_train,arr_test):
    pred_train = []
    pred_test = []

    for i in range(1, len(arr_train)):
        pred_train.append(arr_train[(i-1)])

    for j in range(len(arr_test)):
        # print(j)
        pred_test.append(arr_train[-1])

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)
    residual_err = arr_train[1:] - pred_train
    forecast_err = arr_test - pred_test

    return pred_train, pred_test, residual_err, forecast_err

def drift_method(arr_train, arr_test):
    pred_train = []
    pred_test = []
    val1 = 0

    for i in range(2, len(arr_train)):
        val = arr_train[i - 1] + ((1) * ((arr_train[i - 1] - arr_train[0]) / (i - 1)))
        # val = arr_train[i] + (i + 1) * ((arr_train[i] - arr_train[0]) / i)
        pred_train.append(val)

    for j in range(len(arr_test)):
        val1 = arr_train[-1] + (j + 1) * ((arr_train[-1] - arr_train[0]) / (len(arr_train) - 1))
        pred_test.append(val1)

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)

    residual_err = arr_train[2:] - pred_train
    forecast_err = arr_test - pred_test

    return pred_train, pred_test, residual_err, forecast_err


def ses_method(arr_train,arr_test,alpha):

    pred_train = []
    pred_test = []
    val = 0
    val1 = 0

    for i in range(0, len(arr_train)):
        if i < 1:
            pred_train.append(arr_train[0])
        else:
            val = (alpha * arr_train[i-1] ) + ((1 - alpha)*pred_train[i-1])
            pred_train.append(val)

    for j in range(len(arr_test)):
        val1 = (alpha * arr_train[-1] ) + ((1 - alpha)*pred_train[-1])
        pred_test.append(val1)

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)
    residual_err = arr_train - pred_train
    forecast_err = arr_test - pred_test

    return pred_train, pred_test, residual_err, forecast_err

#############################################################################################################
# Average Method
#############################################################################################################


pred_train, pred_test, residual_error, forecast_error = average_forecast_method(df_train_temp.values, df_test_temp.values)
pred_train = pd.DataFrame(pred_train[:len(df_train_temp)-1], index=df_train_temp.index[:-1])
pred_train = pred_train.sort_index()
pred_test = pd.DataFrame(pred_test[:len(df_test_temp)], index=df_test_temp.index)
pred_test = pred_test.sort_index()


plot_model_results(df_train_temp, df_test_temp, pred_test, pred_train, 'Average Method')
plot_graph(residual_error, 'Average Method')
average1 = calculate_tests(residual_error, forecast_error, df_train_temp, 'Average Method')

df_train_temp1, df_test_temp1 = train_test_split(df_weather['temperature'], test_size=0.2)
pred_train1, pred_test1, residual_error1, forecast_error1 = average_forecast_method(df_train_temp1.values, df_test_temp1.values)
plot_graph(residual_error1, 'Average Method')

#############################################################################################################
# Naive Method
#############################################################################################################


pred_train, pred_test, residual_error, forecast_error = naive_method(df_train_temp.values, df_test_temp.values)
pred_train = pd.DataFrame(pred_train[:len(df_train_temp)-1], index=df_train_temp.index[:-1])
pred_train = pred_train.sort_index()
pred_test = pd.DataFrame(pred_test[:len(df_test_temp)], index=df_test_temp.index)
pred_test = pred_test.sort_index()


plot_model_results(df_train_temp, df_test_temp, pred_test, pred_train, 'Naive Method')
plot_graph(residual_error, 'Naive Method')
naive1 = calculate_tests(residual_error, forecast_error, df_train_temp, 'Naive Method')

df_train_temp1, df_test_temp1 = train_test_split(df_weather['temperature'], test_size=0.2)
pred_train1, pred_test1, residual_error1, forecast_error1 = naive_method(df_train_temp1.values, df_test_temp1.values)
plot_graph(residual_error1, 'Naive Method')


#############################################################################################################
# Drift Method
#############################################################################################################


pred_train, pred_test, residual_error, forecast_error = drift_method(df_train_temp.values, df_test_temp.values)
pred_train = pd.DataFrame(pred_train, index=df_train_temp.index[:len(pred_train)])
pred_test = pd.DataFrame(pred_test, index=df_test_temp.index[:len(pred_test)])
pred_train = pred_train.sort_index()
pred_test = pred_test.sort_index()

plot_model_results(df_train_temp, df_test_temp, pred_test, pred_train, 'Drift Method')
plot_graph(residual_error, 'Drift Method')
drift1 = calculate_tests(residual_error, forecast_error, df_train_temp, 'Drift Method')

df_train_temp1, df_test_temp1 = train_test_split(df_weather['temperature'], test_size=0.2)
pred_train1, pred_test1, residual_error1, forecast_error1 = drift_method(df_train_temp1.values, df_test_temp1.values)
plot_graph(residual_error1, 'Drift Method')

#############################################################################################################
# Simple Exponential Smoothing Method(SES)
#############################################################################################################

pred_train, pred_test, residual_error, forecast_error = ses_method(df_train_temp.values, df_test_temp.values, 0.5)
pred_train = pd.DataFrame(pred_train, index=df_train_temp.index[:len(pred_train)])
pred_test = pd.DataFrame(pred_test, index=df_test_temp.index[:len(pred_test)])
pred_train = pred_train.sort_index()
pred_test = pred_test.sort_index()

plot_model_results(df_train_temp, df_test_temp, pred_test, pred_train, 'SES Method')
plot_graph(residual_error, 'SES Method')
ses1 = calculate_tests(residual_error, forecast_error, df_train_temp, 'SES Method')

df_train_temp1, df_test_temp1 = train_test_split(df_weather['temperature'], test_size=0.2)
pred_train1, pred_test1, residual_error1, forecast_error1 = ses_method(df_train_temp1.values, df_test_temp1.values, 0.5)
plot_graph(residual_error1, 'SES Method')


#############################################################################################################
# Order Estimation
#############################################################################################################

# GPAC
def AutoCorrelation(y, tau):
    y = [x for x in y if np.isnan(x) == False]
    y_bar = np.mean(y)
    numerator = np.sum((y[tau:] - y_bar) * (y[:len(y) - tau] - y_bar))
    denominator = sum((y - y_bar) ** 2)
    result = numerator / denominator
    return result

def ACF_parameter(series, lag=None, removeNA=False, two_sided=False):
    T = len(series)
    if removeNA:
        series = [x for x in series if np.isnan(x) == False]
    else:
        series = list(series)
    if lag is None:
        lag = min(int(10 * np.log10(T)), T - 1)
    res = []
    for i in np.arange(0, lag + 1):
        res.append(AutoCorrelation(series, i))
    if two_sided:
        res = np.concatenate((np.reshape(res[::-1], lag + 1), res[1:]))
    else:
        res = np.array(res)
    return res

def GPAC_cal(series, lags, Lj, Lk, ry_2=None, returntable=False, GPAC_title='Generalized Partial AutoCorrelation(GPAC) Table'):
    if ry_2 is not None:
        if not np.array_equal(ry_2, ry_2[::-1]):
            ry_2 = np.concatenate((np.reshape(ry_2[::-1], len(ry_2)), ry_2[1:]))
    else:
        ry_2 = ACF_parameter(series, lags, two_sided=True)

    ry0 = int((len(ry_2) - 1) / 2)

    if min(Lk, Lj) <= 3:
        raise Exception('Length of the table is recommended to be at least 4')

    table = []
    for j in range(Lj):
        newrow = []
        for k in range(1, Lk + 1):

            num = np.array([]).reshape(k, 0)
            for p in range(k):
                if p != k - 1:
                    newcol = []
                    for q in range(k):
                        newcol.append([ry_2[ry0 + j + q - p]])
                    num = np.hstack((num, newcol))
                else:
                    newcol = []
                    for q in range(k):
                        newcol.append([ry_2[ry0 + 1 + j + q]])
                    num = np.hstack((num, newcol))

            den = np.array([]).reshape(k, 0)
            for p in range(k):
                newcol = []
                for q in range(k):
                    newcol.append([ry_2[ry0 + j + q - p]])
                den = np.hstack((den, newcol))

            # Cramer's Rule
            phi = np.round(np.linalg.det(num) / np.linalg.det(den), 3)
            newrow.append(phi)
        table.append(newrow)

    table = pd.DataFrame(table)
    table.columns = [str(x) for x in range(1, Lk + 1)]

    sns.heatmap(table, annot=True, fmt=".3f", vmin=np.quantile(table, .05), vmax=np.quantile(table, .95))
    plt.title(f"{GPAC_title}")
    plt.tight_layout()
    plt.show()

    print("The GPAC table value is:")
    print(table)
    if returntable is True:
        return table
    else:
        return None

GPAC_cal(new_df_ss_1['Vancouver_2'], lags=50, Lj=7, Lk=7)

GPAC_cal(new_df_ss_1['Vancouver_2'], lags=50, Lj=10, Lk=10)


########################################################################################################################
# LM Method

na=1
nb=0
n = na + nb
lr = 10 ** -6
ep = 10 ** -3
it = 100
mu_max = 10 ** 20

def residual(t, na, y):
    np.random.seed(6313)
    num = t[na:]
    den = t[:na]
    if len(den) > len(num):
        for i in range(len(den) - len(num)):
            num = np.append(num, 0)
    if len(den) < len(num):
        for i in range(len(num) - len(den)):
            den = np.append(den, 0)
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    system = (den, num, 1)
    _, e = signal.dlsim(system, y)
    return e

def roots(theta, na):
    x = theta
    den = x[:na] #err
    num = x[na:] #y

    if len(den) > len(num):
        diff = len(den)-len(num)
        num = np.pad(num, (0,diff), 'constant', constant_values=0)

    elif len(num) > len(den):
        diff = len(num)-len(den)
        den = np.pad(den, (0,diff), 'constant', constant_values=0)

    den = np.r_[1, den]
    num = np.r_[1, num]

    a=np.roots(num)
    b=np.roots(den)
    print('Poles and zeros')
    print(a)
    print(b)

def lm1(t, na, y):
    np.random.seed(6313)
    e = residual(t, na, y)
    sset = e.T @ e
    X = np.empty([len(y), n])
    for i in range(0, n):
        t[i] = t[i] + lr
        e_i = residual(t, na, y)
        x_i = (e - e_i) / lr
        X[:, i] = x_i[:, 0]
        t[i] = t[i] - lr
    a = X.T @ X
    g = X.T @ e
    return sset, X, a, g


def lm2(t, mu, a, g, na, y):
    np.random.seed(6313)
    change = np.linalg.inv(a + mu * np.identity(n)) @ g
    new_t = t + change
    e_new = residual(new_t, na, y)
    new_sse = e_new.T @ e_new
    return change, new_t, new_sse


def lm3_final(na, y):
    np.random.seed(6313)
    theta = np.zeros([n, 1])
    mu = 0.01
    var_e = 0
    cov_theta_hat = 0
    SSE_list = []
    for i in range(it):
        SSE_theta, X, A, g = lm1(theta, na, y)
        theta_change, theta_new, SSE_theta_new = lm2(theta, mu, A, g, na, y)
        SSE_list.append(SSE_theta[0][0])
        if i < it:
            if SSE_theta_new < SSE_theta:
                if np.linalg.norm(np.array(theta_change), 2) < ep:
                    theta_hat = theta_new
                    var_e = SSE_theta_new / (len(y) - n)
                    cov_theta_hat = var_e * np.linalg.inv(A)
                    break
                else:
                    theta = theta_new
                    mu = mu / 10
            while SSE_theta_new >= SSE_theta:
                mu = mu * 10
                if mu > mu_max:
                    print('No results')
                    break
                theta_change, theta_new, SSE_theta_new = lm2(theta, mu, A, g, na, y)
        if i > it:
            print('Iternations completed and no results')
            break
        theta = theta_new
    return theta_new, SSE_theta_new, var_e, cov_theta_hat, SSE_list

teta, sse, varianc, cov_matrix, sse_list = lm3_final(na, new_df_ss_1['Vancouver_2'])
print('The estimated parameters are: ', teta)

def display_ci(theta, covariance_matrix):
    # Compute diagonal elements of covariance matrix
    np.random.seed(6313)
    diagonal = covariance_matrix.diagonal()
    # Compute the interval bounds
    a = 2 * np.sqrt(diagonal)
    upper_bound = theta + a
    lower_bound = theta - a
    # Print the confidence interval bounds
    print("Upper bound: ")
    print(upper_bound)
    print("Lower bound: ")
    print(lower_bound)

display_ci(np.around(teta, 3), np.around(cov_matrix, 3))


########################################################################################################################

########################################################################################################################

# Since data is seasonal we are going ot use sarima model

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# def arima_model(na, nb, d, lags, y_train, y_test):
#     # Create ARIMA model
#     model = ARIMA(y_train, order=(na, d, nb))
#     # Fit the model
#     model_fit = model.fit()
#     # Make predictions
#     pred = model_fit.forecast(steps=len(y_test))
#     # Calculate the mean squared error
#     mse = ((pred - y_test) ** 2).mean()
#     # Create a DataFrame to store predictions
#     pred_df = pd.DataFrame({'predictions': pred}, index=y_test.index)
#     # Add the actual values to the DataFrame
#     pred_df['actual'] = y_test.values
#     # Add the lagged values to the DataFrame
#     for i in range(1, lags + 1):
#         pred_df[f'lag_{i}'] = y_test.shift(i).values
#      # Perform F-test
#     f_test = model_fit.f_test(np.identity(len(model_fit.params))[:na+nb+1])
#     # Print F-test results
#     print("F-Test Results:")
#     print(f"p-value: {f_test.pvalue}")
#     print(f"F-Statistic: {f_test.fvalue}")
#      # Perform T-test
#     t_test = model_fit.t_test(np.identity(len(model_fit.params))[0])
#     # Print T-test results
#     print("\nT-Test Results:")
#     print(f"p-value: {t_test.pvalue[0][0]}")
#     print(f"T-Statistic: {t_test.tvalue[0][0]}")
#      # Perform Chi-square test
#     lbvalue, pvalue = acorr_ljungbox(model_fit.resid, lags=[10])
#     # Print Chi-square test results
#     print("\nChi-Square Test Results:")
#     print(f"p-value: {pvalue[0]}")
#     print(f"Chi-Square Statistic: {lbvalue[0]}")
#      # Print AIC and BIC
#     print(f"\nAIC: {model_fit.aic}")
#     print(f"BIC: {model_fit.bic}")
#      # Print R-squared
#     print(f"\nR-Squared: {model_fit.rsquared}")
#      # Calculate and print RMSE
#     rmse = np.sqrt(mse)
#     print(f"\nRMSE: {rmse}")
#      # Calculate and print adjusted R-squared
#     n = len(y_train) + len(y_test)
#     k = na + nb + 1
#     adj_r_squared = 1 - (((1 - model_fit.rsquared) * (n - 1)) / (n - k - 1))
#     print(f"\nAdjusted R-Squared: {adj_r_squared}")
#      # Calculate and print variance and mean of residuals
#     residuals = model_fit.resid
#     var_res = residuals.var()
#     mean_res = residuals.mean()
#     print(f"\nVariance of residuals: {var_res}")
#     print(f"Mean of residuals: {mean_res}")
#      # Perform Q-value test
#     q_value, p_value = sm.stats.diagnostic.acorr_ljungbox(model_fit.resid, lags=[lags])
#     print(f"\nQ-value: {q_value[0]}, p-value: {p_value[0]}")
#     return pred_df


############################################################################################################
# SARIMA
############################################################################################################

from statsmodels.tsa.statespace.sarimax import SARIMAX
def SARIMA_modeling(y_train, y_test, na, nb, d, seasonal_order=24):
    """
    Performs the SARIMA modeling on the given training set.

    Args:
        y_train (numpy.ndarray): The training set.
        y_test (numpy.ndarray): The test set.
        na (int): The AR order.
        nb (int): The MA order.
        d (int): The difference order.
        seasonal_order (tuple): The seasonal order (P, D, Q, S).

    Returns:
        tuple: A tuple containing the SARIMA model, predicted values for the training set, predicted values for the test set, residual error, and forecast error.

    """

    # Create the SARIMA model.
    model = SARIMAX(y_train, order=(na, d, nb), seasonal_order=seasonal_order).fit()

    # Print the summary of the model.
    print(model.summary())

    # Calculate the predicted values for the training set.
    pred_train = model.predict(start=0, end=len(y_train)-1)

    # Calculate the predicted values for the test set.
    pred_test = model.forecast(steps=len(y_test))

    # Calculate the residual error.
    residual_err = y_train - pred_train

    # Calculate the forecast error.
    forecast_err = y_test - pred_test

    return model, pred_train, pred_test, residual_err, forecast_err


def ARIMA_modeling(y_train,y_test, na, nb, d):
    """
    Performs the ARIMA modeling on the given training set.

    Args:
        y_train (numpy.ndarray): The training set.
        na (int): The AR order.
        nb (int): The MA order.
        d (int): The difference order.

    Returns:
        statsmodels.tsa.arima_model.ARIMAResults: The ARIMA model.

    """

    # Create the ARIMA model.
    model = ARIMA(y_train, order=(na, d, nb)).fit()

    # Print the summary of the model.
    print(model.summary())

    # Calculate the predicted values for the training set.
    pred_train = model.predict(start=1, end=len(y_train)-1)

    # Calculate the predicted values for the test set.
    pred_test = model.forecast(steps=len(y_test))

    # Calculate the residual error.
    residual_err = y_train - pred_train

    # Calculate the forecast error.
    forecast_err = y_test - pred_test

    return model, pred_train, pred_test, residual_err, forecast_err

def AR_MA_graphs(model, y_train, y_test):
    """
    Plots the graphs for the given ARIMA model.

    Args:
        model (statsmodels.tsa.arima_model.ARIMAResults): The ARIMA model.
        y_train (numpy.ndarray): The training set.
        y_test (numpy.ndarray): The test set.

    """

    # Plot the diagnostic plots.
    model.plot_diagnostics(figsize=(14, 10))
    plt.suptitle('ARIMA Diagnostic Analysis')
    plt.grid()
    plt.show()

    # Plot the ACF graph.
    # plt.figure()
    # acf = sm.tsa.stattools.acf(model.resid, lags=50)
    # plt.plot(acf)
    # plt.axhline(0, color='black')
    # plt.axhline(1.96 / np.sqrt(len(y_train)), color='red', linestyle='--')
    # plt.axhline(-1.96 / np.sqrt(len(y_train)), color='red', linestyle='--')
    # plt.title('ACF of ARIMA residuals')
    # plt.grid()
    # plt.show()
    pred_train = model.predict()
    # pred_train = pd.DataFrame(pred_train, index=df_train_temp.index)
    # pred_train = pred_train.sort_index()
    # Plot the train data vs fitted data graph.
    plt.figure()
    plt.plot(y_train, label='Train data')
    plt.plot(model.predict(), label='Fitted values')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('ARIMA model and predictions')
    plt.grid()
    plt.show()

    # Plot the test data vs forecasted values graph.
    plt.figure()
    plt.plot(y_test, label='Test data')
    plt.plot(model.forecast(len(y_test)), label='Forecasted values')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('ARIMA model and forecast')
    plt.grid()
    plt.show()


def autocorrelation(lags,list):
    val = [1]
    lag = [0]
    den = 0
    num = 0
    for k in range (0,len(list)):
        d = (list[k] - list.mean())**2
        den+= d
    for i in range(1,lags+1):
        num = 0
        for j in range (i,len(list)):
            n = ((list[j] - list.mean()) * (list[abs(j-i)]- list.mean()))
            num+= n
        cor = num/den
        val+=[cor]
        lag += [i]
    lag = np.array(lag)
    val = np.array(val)
    vals = val
    vals = vals[vals != 1]
    val1 = val[1:]
    lag1 = np.negative(lag[1:])
    val = np.concatenate((val1[::-1],val), axis=None)
    lag = np.concatenate((lag1[::-1], lag), axis=None)
    val = val[val != -1]
    return val,lag,vals
def plot_graph(residual_err, title):
    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation for ' + title)
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

def sarima_model(train,test, na,nb,d,seasonal_order):
    '''
    train: training set
    test: test set
    na: AR order
    nb: MA order
    d: difference order
    seasonal_order: seasonal order (P,D,Q,S)
    Returns:
        tuple: A tuple containing the SARIMA model, predicted values for the training set, predicted values for the test set, residual error, and forecast error.
    '''
    model = SARIMAX(train, order=(na,d,nb), seasonal_order=seasonal_order).fit()
    print(model.summary())
    pred_train = model.predict(start=1, end=len(train)-1)
    pred_test = model.forecast(steps=len(test))
    residual_err = train - pred_train
    forecast_err = test - pred_test
    return model, pred_train, pred_test, residual_err, forecast_err
def AR_MA_metrics(model,residual_err, forecast_err, y_train, y_test):
    """
    Computes and prints the metrics for the given ARIMA model.

    Args:
        model (statsmodels.tsa.arima_model.ARIMAResults): The ARIMA model.
        y_train (numpy.ndarray): The training set.
        y_test (numpy.ndarray): The test set.

    """

    # Compute the AIC for the model.
    AIC = model.aic

    # Compute the BIC for the model.
    BIC = model.bic
    # Compute the F-test for the model.
    F_statistic, p_value = stats.f_oneway(residual_err, forecast_err)

    # Compute the Q-value for the model.
    Q = sm.stats.acorr_ljungbox(residual_err, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]

    # Compute the RMSE for the training set.
    mse_train = np.mean((y_train - model.predict()) ** 2)

    # Compute the RMSE for the test set.
    mse_test = np.mean((y_test - model.forecast(len(y_test))) ** 2)

    # Compute the R^2 for the training set.
    r_squared_train = 1 - np.var(y_train - model.predict()) / np.var(y_train)

    # Compute the R^2 for the test set.
    r_squared_test = 1 - np.var(y_test - model.forecast(len(y_test))) / np.var(y_test)

    # Print the metrics.
    print('AIC: {}'.format(AIC))
    print('BIC: {}'.format(BIC))
    print('F-statistic: {}'.format(F_statistic))
    print('p-value: {}'.format(p_value))
    print('Q-value: {}'.format(Q))
    print('MSE for training set: {}'.format(mse_train))
    print('MSE for test set: {}'.format(mse_test))
    print('R^2 for training set: {}'.format(r_squared_train))
    print('R^2 for test set: {}'.format(r_squared_test))

split_index = int(len(df_weather) * 0.8)  # Index to split at (80% of the data)
df_train_temp = df_weather['temperature'][:split_index]  # First 80% as training
df_test_temp = df_weather['temperature'][split_index:]  # Last 20% as testing


# Order: ar = 1, d = 1, ma = 0
na=1
nb=0
d=1
model, pred_train, pred_test, residual_err, forecast_err = sarima_model(df_train_temp, df_test_temp,na, nb, d, (0, 1, 1, 24))
pred_train = pd.DataFrame(pred_train, index=df_train_temp.index)
pred_train = pred_train.sort_index()
pred_test = pd.DataFrame(pred_test, index=df_test_temp.index)
pred_test = pred_test.sort_index()
pred_train = pred_train.fillna(pred_train.mean())
pred_test = pred_test.fillna(pred_test.mean())
mean_residual = np.mean(residual_err)
residual_err[np.isnan(residual_err)] = mean_residual
mean_forecast = np.mean(forecast_err)
forecast_err[np.isnan(forecast_err)] = mean_forecast

AR_MA_metrics(model,residual_err, forecast_err, df_train_temp, df_test_temp)

title = f'SARIMA (na={na}, nb={nb}, d={d} and (P,D,Q,S)=(0,0,1,24))'
plt.figsize=(15, 15)
plt.plot(df_train_temp, label='Train')
plt.plot(pred_train, label='Predicted Data')
plt.legend()
plt.title(f'Train vs Predicted for {title}')
plt.xlabel('Samples')
plt.ylabel('Temperature')
plt.tight_layout()
plt.show()

plt.plot(df_test_temp, label='Test')
plt.plot(pred_test, label='Forecasted Data')
plt.legend()
plt.title(f'Test vs Forecasted for {title}')
plt.xlabel('Samples')
plt.ylabel('Temperature')
plt.tight_layout()
plt.show()

plot_graph(residual_err.values, title)


# order: ar=1, ma=3, d=1
na=1
nb=3
d=1
model, pred_train, pred_test, residual_err, forecast_err = sarima_model(df_train_temp, df_test_temp,na, nb, d, (0, 1, 1, 24))
pred_train = pd.DataFrame(pred_train, index=df_train_temp.index)
pred_train = pred_train.sort_index()
pred_test = pd.DataFrame(pred_test, index=df_test_temp.index)
pred_test = pred_test.sort_index()
pred_train = pred_train.fillna(pred_train.mean())
pred_test = pred_test.fillna(pred_test.mean())
mean_residual = np.mean(residual_err)
residual_err[np.isnan(residual_err)] = mean_residual
mean_forecast = np.mean(forecast_err)
forecast_err[np.isnan(forecast_err)] = mean_forecast

AR_MA_metrics(model,residual_err, forecast_err, df_train_temp, df_test_temp)

title = f'SARIMA (na={na}, nb={nb}, d={d} and (P,D,Q,S)=(0,0,1,24))'
plt.figsize=(15, 15)
plt.plot(df_train_temp, label='Train')
plt.plot(pred_train, label='Predicted Data')
plt.legend()
plt.title(f'Train vs Predicted for {title}')
plt.xlabel('Samples')
plt.ylabel('Temperature')
plt.tight_layout()
plt.show()

plt.plot(df_test_temp, label='Test')
plt.plot(pred_test, label='Forecasted Data')
plt.legend()
plt.title(f'Test vs Forecasted for {title}')
plt.xlabel('Samples')
plt.ylabel('Temperature')
plt.tight_layout()
plt.show()

plot_graph(residual_err.values, title)

import numpy as np

def find_best_order(data, test_data):
    """
    Finds the best order for an ARIMA model on the given data.

    Args:
        data (numpy.ndarray): The data to train the model on.
        test_data (numpy.ndarray): The data to test the model on.
        metric (str): The metric to use to evaluate the models.

    Returns:
        A dictionary with the top 5 combination and their respective adjusted R^2.
    """

    # Get the possible orders.

    orders = [(p, d, q) for p in range(1, 5) for d in range(0, 2) for q in range(1, 5)]

    # Initialize the best order and the best metric value.
    best_order = (0, 0, 0)
    best_metric_value = float('-inf')

    # Create a dictionary to store the results.
    results = {}

    # Loop over the orders.
    for order in orders:
        # Train the model on the given order.
        model = ARIMA(data, order=order).fit()

        # Get the predictions for the model.
        predictions = model.predict(len(data))

        # Get the residuals for the model.
        residuals = test_data - predictions
        p=order[0]
        d=order[1]
        q=order[2]
        # Calculate the adjusted R^2 for the model.
        adjusted_r2_value = 1 - (1 - np.mean(residuals ** 2)) * (len(data) - 1) / (len(data) - p - d - q - 1)

        # Add the results to the dictionary.
        results[order] = adjusted_r2_value

        # If the adjusted R^2 value is better than the best adjusted R^2 value, update the best order and the best metric value.
        if adjusted_r2_value > best_metric_value:
            best_order = order
            best_metric_value = adjusted_r2_value

    # Return the dictionary with the top 5 combination and their respective adjusted R^2.
    return {k: v for k, v in sorted(results.items(), key=lambda x: x[1])[:5]}

# a=find_best_order(df_train_temp.values, df_test_temp.values)
# print(a)














