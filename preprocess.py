#%%
import pandas as pd

oecd_housing = pd.read_csv('data/oecd_housing_prices.csv')
# we're only interested in "real" prices
oecd_housing = oecd_housing[oecd_housing['SUBJECT'] == 'REAL']
# we're only interested annual data
oecd_housing = oecd_housing[oecd_housing['FREQUENCY'] == 'A']
oecd_housing
# %%
countries = oecd_housing['LOCATION'].unique()
countries

# %%
oecd_housing[oecd_housing['LOCATION'] == 'HUN'].plot(x = "TIME", y= "Value")
# %%

# %%
