#!/usr/bin/env python3
"""
This code runs an expanding window out-of-sample forecasting exercise.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import multiprocessing as mp
import os
import time
import statsmodels.api as sm
from scipy.stats import t as tstat
from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import shutil


# User Functions

def process_data(filename, start_date=None, end_date=None):
    # 读取数据
    df = pd.read_excel(filename)

    # 将第一列设置为时间索引并转换为日期类型
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], format='%Y%m')
    df.set_index('Unnamed: 0', inplace=True)

    # 如果提供了起止日期，则截取该时间段的数据
    if start_date and end_date:
        df = df.loc[pd.to_datetime(start_date, format='%Y%m'):pd.to_datetime(end_date, format='%Y%m')]

    # 读取Y数组（以xr_开头的变量）
    Y = df.filter(regex='^xr_').values

    # 读取Xexog数组（以f_开头的变量）
    Xexog = df.filter(regex='^f_').values

    # 读取Yield数组（列名以m结尾的数据）
    Yield = df.filter(regex='m$').values

    # 读取X数组（其他变量）
    X = df.drop(columns=df.filter(regex='(^xr_)|(^f_)|m$').columns).values

    # 返回时间索引列
    time = df.index

    return X, Y, Xexog, Yield, time


def multProcessOwnExog(func, ncpus, nMC, X, Xexog, Y, **kwargs):
    try:
        pool = mp.Pool(processes=ncpus)
        output = [pool.apply_async(func, args=(X, Xexog, Y, no,), kwds=kwargs) for no in range(nMC)]
        outputCons = [p.get(timeout=3000) for p in output]
        pool.close()
        pool.join()
        time.sleep(1)
    except Exception as e:
        print(e)
        print("Timed out, shutting pool down")
        pool.close()
        pool.terminate()
        time.sleep(1)
    return outputCons


def R2OOS(y_true, y_forecast):
    import numpy as np

    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size) + 1))
    y_condmean = np.insert(y_condmean, 0, np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    SSres = np.nansum(np.square(y_true - y_forecast))
    SStot = np.nansum(np.square(y_true - y_condmean))

    return 1 - SSres / SStot


def RSZ_Signif(y_true, y_forecast):
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size) + 1))
    y_condmean = np.insert(y_condmean, 0, np.nan)
    y_condmean = y_condmean[:-1]
    y_condmean[np.isnan(y_forecast)] = np.nan

    f = np.square(y_true - y_condmean) - np.square(y_true - y_forecast) + np.square(y_condmean - y_forecast)
    x = np.ones(np.shape(f))
    model = sm.OLS(f, x, missing='drop', hasconst=True)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    return 1 - tstat.cdf(results.tvalues[0], results.nobs - 1)


def ElasticNet_Exog_Plain(X, Xexog, Y):
    X_train = X[:-1, :]
    Xexog_train = Xexog[:-1, :]
    Y_train = Y[:-1, :]
    X_test = X[-1, :].reshape(1, -1)
    Xexog_test = Xexog[-1, :].reshape(1, -1)

    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)

    N_train = int(np.round(np.size(X_train, axis=0) * 0.85))
    N_val = np.size(X_train, axis=0) - N_train
    test_fold = np.concatenate(((np.full((N_train), -1), np.full((N_val), 0))))
    ps = PredefinedSplit(test_fold.tolist())

    Ypred = np.full([1, Y_train.shape[1]], np.nan)
    for i in range(Y_train.shape[1]):
        model = ElasticNetCV(cv=ps, max_iter=5000, n_jobs=-1, l1_ratio=[.1, .3, .5, .7, .9], random_state=42)
        model = model.fit(np.concatenate((X_train, Xexog_train), axis=1), Y_train[:, i])
        Ypred[0, i] = model.predict(np.concatenate((X_test, Xexog_test), axis=1))

    return Ypred


def PCA_Exog(X, Xexog, Y, n_components_X, m_components_Xexog):
    scaler_X = StandardScaler()
    scaler_Xexog = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Xexog_scaled = scaler_X.fit_transform(Xexog)

    pca_X = PCA(n_components=n_components_X)
    pca_Xexog = PCA(n_components=m_components_Xexog)

    X_train_pca = pca_X.fit_transform(X_scaled[:-1, :])
    Xexog_train_pca = pca_Xexog.fit_transform(Xexog_scaled[:-1, :])

    X_test_pca = pca_X.transform(X_scaled[-1, :].reshape(1, -1))
    Xexog_test_pca = pca_Xexog.transform(Xexog_scaled[-1, :].reshape(1, -1))

    X_combined_train = np.concatenate((X_train_pca, Xexog_train_pca), axis=1)
    X_combined_test = np.concatenate((X_test_pca, Xexog_test_pca), axis=1)

    model = LinearRegression()
    model.fit(X_combined_train, Y[:-1, :])
    Ypred = model.predict(X_combined_test)

    return Ypred


def SimplePLS_Exog(X, Xexog, Y, n_components):
    scaler = StandardScaler()
    X_total_scaled = scaler.fit_transform(np.concatenate((X, Xexog), axis=1))

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_total_scaled[:-1, :], Y[:-1, :])
    Ypred = pls.predict(X_total_scaled[-1, :].reshape(1, -1))

    return Ypred


def custom_PLS_Exog(X, Xexog, Y):
    scaler = StandardScaler()
    X_total_scaled = scaler.fit_transform(np.concatenate((X, Xexog), axis=1))

    coefficients = np.zeros(X_total_scaled.shape[1])
    for j in range(X_total_scaled.shape[1]):
        model = LinearRegression()
        model.fit(X_total_scaled[:-1, j:j + 1], Y[:-1, :])
        coefficients[j] = model.coef_[0][0]

    x1 = X_total_scaled @ coefficients.T
    x1 = x1.reshape(-1, 1)

    model_x1 = LinearRegression()
    model_x1.fit(x1[:-1, :], Y[:-1, :])
    Ypred = model_x1.predict(x1[-1, :].reshape(1, -1))

    return Ypred


def Lasso_Exog(X, Xexog, Y):
    X_train = X[:-1, :]
    Xexog_train = Xexog[:-1, :]
    Y_train = Y[:-1, :]
    X_test = X[-1, :].reshape(1, -1)
    Xexog_test = Xexog[-1, :].reshape(1, -1)

    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)

    N_train = int(np.round(np.size(X_train, axis=0) * 0.85))
    N_val = np.size(X_train, axis=0) - N_train
    test_fold = np.concatenate(((np.full((N_train), -1), np.full((N_val), 0))))
    ps = PredefinedSplit(test_fold.tolist())

    Ypred = np.full([1, Y_train.shape[1]], np.nan)
    for i in range(Y_train.shape[1]):
        model = LassoCV(n_jobs=-1, cv=ps, max_iter=5000, random_state=42)
        model = model.fit(np.concatenate((X_train, Xexog_train), axis=1), Y_train[:, i])
        Ypred[0, i] = model.predict(np.concatenate((X_test, Xexog_test), axis=1))

    return Ypred


def Ridge_Exog(X, Xexog, Y):
    X_train = X[:-1, :]
    Xexog_train = Xexog[:-1, :]
    Y_train = Y[:-1, :]
    X_test = X[-1, :].reshape(1, -1)
    Xexog_test = Xexog[-1, :].reshape(1, -1)

    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)

    N_train = int(np.round(np.size(X_train, axis=0) * 0.85))
    N_val = np.size(X_train, axis=0) - N_train
    test_fold = np.concatenate(((np.full((N_train), -1), np.full((N_val), 0))))
    ps = PredefinedSplit(test_fold.tolist())

    Ypred = np.full([1, Y_train.shape[1]], np.nan)
    for i in range(Y_train.shape[1]):
        model = RidgeCV(cv=ps)
        model = model.fit(np.concatenate((X_train, Xexog_train), axis=1), Y_train[:, i])
        Ypred[0, i] = model.predict(np.concatenate((X_test, Xexog_test), axis=1))

    return Ypred


if __name__ == "__main__":

    # =========================================================================
    #                           Settings
    # =========================================================================

    TestFlag = False
    if TestFlag:
        nMC = 4
        nAvg = 2
    else:
        nMC = 100
        nAvg = 10

    OOS_Start = '1990-01-01'
    data_path = './test'

    HyperFreq = 4 * 12

    ncpus = mp.cpu_count()
    print("CPU count is: " + str(ncpus))

    dumploc_base = './test/trainingDumps_'

    i = 0
    path_established = False
    while not path_established:
        dumploc = dumploc_base + str(i)
        try:
            os.makedirs(dumploc)
            print("Directory ", dumploc, " Created ")
            path_established = True
        except FileExistsError:
            print("Directory ", dumploc, " Already exists")
            i += 1

    models = [PCA_Exog, Ridge_Exog, SimplePLS_Exog, custom_PLS_Exog, Lasso_Exog]
    modelnames = ['PCA_Exog', 'Ridge_Exog', 'SimplePLS_Exog', 'custom_PLS_Exog', 'Lasso_Exog']

    X, Y, Xexog, Yield, time_data = process_data('./data.xlsx', start_date='197108', end_date='202112')
    X = X[:, :10]

    T = X.shape[0]
    tstart = np.argmax(time_data == OOS_Start)
    OoS_indeces = range(tstart, int(T))
    print(OoS_indeces)
    M = Y.shape[1]

    if TestFlag:
        OoS_indeces = OoS_indeces[:2]

    VarSave = {}

    for modelnum, modelfunc in enumerate(models):
        Y_forecast = np.full([T, nMC, M], np.nan)
        Y_forecast_agg = np.full([T, M], np.nan)
        val_loss = np.full([T, nMC], np.nan)
        print(modelnames[modelnum])

        if modelnames[modelnum] == 'PCA_Exog':
            n_components_X = 4
            m_components_Xexog = 3

            for i in OoS_indeces[:-10]:
                start = time.time()
                ypredmean = modelfunc(X[:i + 1, :], Xexog[:i + 1, :], Y[:i + 1, :], n_components_X, m_components_Xexog)
                Y_forecast[i, 0, :] = ypredmean
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        elif modelnames[modelnum] == 'SimplePLS_Exog':
            n_components = 8

            for i in OoS_indeces[:-10]:
                start = time.time()
                ypredmean = modelfunc(X[:i + 1, :], Xexog[:i + 1, :], Y[:i + 1, :], n_components)
                Y_forecast[i, 0, :] = ypredmean
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        elif modelnames[modelnum] == 'custom_PLS_Exog':
            for i in OoS_indeces[:-10]:
                start = time.time()
                ypredmean = modelfunc(X[:i + 1, :], Xexog[:i + 1, :], Y[:i + 1, :])
                Y_forecast[i, 0, :] = ypredmean
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        elif modelnames[modelnum] == 'Lasso_Exog':
            for i in OoS_indeces[:-10]:
                start = time.time()
                ypredmean = modelfunc(X[:i + 1, :], Xexog[:i + 1, :], Y[:i + 1, :])
                Y_forecast[i, 0, :] = ypredmean
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        elif modelnames[modelnum] == 'Ridge_Exog':
            for i in OoS_indeces[:-10]:
                start = time.time()
                ypredmean = modelfunc(X[:i + 1, :], Xexog[:i + 1, :], Y[:i + 1, :])
                Y_forecast[i, 0, :] = ypredmean
                Y_forecast_agg[i, :] = ypredmean
                print("Obs No.: ", i, " - Step Time: ", time.time() - start)

        else:
            raise Exception("Model does not match any known case.")

        VarSave["ValLoss_" + modelnames[modelnum]] = val_loss
        VarSave["Y_forecast_agg_" + modelnames[modelnum]] = Y_forecast_agg
        VarSave["Y_forecast_" + modelnames[modelnum]] = Y_forecast
        VarSave["MSE_" + modelnames[modelnum]] = np.nanmean(np.square(Y - Y_forecast_agg), axis=0)
        VarSave["R2OOS_" + modelnames[modelnum]] = np.array([R2OOS(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))])
        print('R2OOS: ', VarSave["R2OOS_" + modelnames[modelnum]])
        VarSave["R2OOS_pval_" + modelnames[modelnum]] = np.array([RSZ_Signif(Y[:, k], Y_forecast_agg[:, k]) for k in range(np.size(Y, axis=1))])

        savesuccess_flag = False
        while not savesuccess_flag:
            try:
                VarSaveSOA = sio.loadmat('ModelComparison_Rolling_SOA.mat')
                VarSaveSOA.update(VarSave)
                sio.savemat('ModelComparison_Rolling_SOA.mat', VarSaveSOA)
                savesuccess_flag = True
                print('Updated SOA file')
            except FileNotFoundError:
                sio.savemat('ModelComparison_Rolling_SOA.mat', VarSave)
                savesuccess_flag = True
                print('Created new SOA file')

    try:
        shutil.rmtree(dumploc)
        print('Removed dir: ' + dumploc + ' succesfully')
    except FileNotFoundError:
        print('Directory: ' + dumploc + ' could not be removed')
