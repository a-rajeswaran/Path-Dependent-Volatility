import pandas as pd


vix = pd.read_csv("data/VIX.csv")
sp = pd.read_csv("data/SPX.csv")

vix["DATE"] = pd.to_datetime(vix["DATE"], format="%m/%d/%Y").dt.date
sp["Date"] = pd.to_datetime(sp["Date"], format="%Y-%m-%d").dt.date

df_vix_spx = pd.merge(vix, sp, left_on='DATE', right_on='Date', how='inner')
df_vix_spx= df_vix_spx.rename(columns={"Close" : "SPX", "CLOSE" : "VIX"})
df_vix_spx = df_vix_spx[["Date", "VIX", "SPX"]]
df_vix_spx['Return'] = (df_vix_spx['SPX'].shift(-1) - df_vix_spx['SPX']) / df_vix_spx['SPX']
df_vix_spx["Return 2"] = df_vix_spx["Return"] * df_vix_spx["Return"]
df_vix_spx.to_excel("data/VIXSPX.xlsx", index=False)


vxn = pd.read_csv("data/VXN.csv")
nas = pd.read_csv("data/NASDAQ.csv")

vxn["DATE"] = pd.to_datetime(vxn["DATE"], format="%m/%d/%Y").dt.date
nas["Date"] = pd.to_datetime(nas["Date"], format="%m/%d/%Y").dt.date

df_vxn_nas = pd.merge(vxn, nas, left_on='DATE', right_on='Date', how='inner')
df_vxn_nas= df_vxn_nas.rename(columns={"Close/Last" : "NASDAQ", "CLOSE" : "VXN"})
df_vxn_nas = df_vxn_nas[["Date", "VXN", "NASDAQ"]]

df_vxn_nas['Return'] = (df_vxn_nas['NASDAQ'].shift(-1) - df_vxn_nas['NASDAQ']) / df_vxn_nas['NASDAQ']
df_vxn_nas["Return 2"] = df_vxn_nas["Return"] * df_vxn_nas["Return"]

df_vxn_nas.to_excel("data/VXNNAS.xlsx", index=False)




vxd = pd.read_excel("data/VXD.xlsx", sheet_name ="Daily, Close")
dowj = pd.read_csv("data/DOW JONES.csv")

vxd["observation_date"] = pd.to_datetime(vxd["observation_date"], format="%Y-%m-%d").dt.date
dowj["Date"] = pd.to_datetime(dowj["Date"], format="%Y-%m-%d").dt.date

df_vxd_dowj = pd.merge(vxd, dowj, left_on='observation_date', right_on='Date', how='inner')
df_vxd_dowj= df_vxd_dowj.rename(columns={"VXDCLS" : "VXD", "Close" : "DOWJ"})
df_vxd_dowj = df_vxd_dowj[["Date", "VXD", "DOWJ"]]

df_vxd_dowj['Return'] = (df_vxd_dowj['DOWJ'].shift(-1) - df_vxd_dowj['DOWJ']) / df_vxd_dowj['DOWJ']
df_vxd_dowj["Return 2"] = df_vxd_dowj["Return"] * df_vxd_dowj["Return"]

df_vxd_dowj.to_excel("data/VXDDOWJ.xlsx", index=False)


print("ok")