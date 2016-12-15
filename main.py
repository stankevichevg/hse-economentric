import sys

import scipy
from arch.univariate import ARX, ConstantMean, GARCH
from scipy.special._ufuncs import errprint
from scipy.stats import chi2
from statsmodels.sandbox.stats.diagnostic import het_arch

import warnings
warnings.simplefilter('ignore')

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')  # Красивые графики
plt.rcParams['figure.figsize'] = (15, 5)  # Размер картинок

mfon_df = pd.read_csv('MFON_121128_161210.csv', parse_dates=['<DATE>'], index_col='<DATE>')
mtss_df = pd.read_csv('MTSS_121128_161210.csv', parse_dates=['<DATE>'], index_col='<DATE>')

mfon_plt = mfon_df['<CLOSE>'].plot(title='MFON dayly close prices')
mtss_plt = mtss_df['<CLOSE>'].plot(title='MTSS dayly close prices')

from arch.unitroot import ADF
adf = ADF(mtss_df['<CLOSE>'])
print(adf.summary().as_text())

adf = ADF(mfon_df['<CLOSE>'])
print(adf.summary().as_text())

mtss_returns = 100 * mtss_df['<CLOSE>'].pct_change().dropna()
mfon_returns = 100 * mfon_df['<CLOSE>'].pct_change().dropna()

mtss_rplt = mtss_returns.plot(title='MTSS dayly returns')
mfon_rplt = mfon_returns.plot(title='MFON dayly returns')

adf = ADF(mtss_returns)
print(adf.summary().as_text())
adf = ADF(mfon_returns)
print(adf.summary().as_text())

from arch import arch_model
mtss_am = arch_model(mtss_returns)
mtss_res = mtss_am.fit(update_freq=5, disp = 'off')
mfon_am = arch_model(mfon_returns)
mfon_res = mfon_am.fit(update_freq=5, disp = 'off')

mfon_res.conditional_volatility
mfon_vol = mfon_res.conditional_volatility * np.sqrt(252)
mtss_res.conditional_volatility
mtss_vol = mtss_res.conditional_volatility * np.sqrt(252)


cm = ConstantMean(mtss_returns)
res = cm.fit(update_freq=5)
f_pvalue = het_arch(res.resid)[3]

cm.volatility = GARCH(p=1, q=1)

p = plt.plot(title='ASSAD')
p1 = plt.plot(mfon_vol)
p2 = plt.plot(mtss_vol)
p = plt.legend((p1[0], p2[0]), ('MFON', 'MTSS'))

from scipy import stats

pvalue = 1 - stats.chi2.cdf(0.940659, 1)

from arch import arch_model
from scipy import stats

def find_garch(values, max_p=5, max_q=5):
    def lr_test(r1, r2):
        return 1 - stats.chi2.cdf(
            abs(2 * (r1.loglikelihood - r2.loglikelihood)),
            abs(r1.num_params - r2.num_params)
        )
    best_model = {"p": -1, "q": -1, "AIC": float('-inf'), "res": 0}
    initialized = False
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            model = arch_model(values, p=p, q=q)
            res = model.fit(disp = 'off')
            if (res.aic > best_model["AIC"]):
                if (initialized):
                    old_res = best_model["res"]
                    best_model = {"p": p, "q": q, "AIC": res.aic, 'res': res}
                    print("Оптимальная модель обновлена: p = %d, q = %d, AIC = %f"
                          % (p, q, res.aic))
                    if (res.num_params != old_res.num_params):
                        print("LR test p-value = %f" % (lr_test(res, old_res)))
                else:
                    best_model = {"p": p, "q": q, "AIC": res.aic, 'res': res}
                    print("Найдена оптимальная модель: p = %d, q = %d, AIC = %f" % (p, q, res.aic))

                initialized = True
    return best_model

mfon_res.forecast()

res= mfon_res

np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])

find_garch(mfon_returns)
find_garch(mtss_returns)
