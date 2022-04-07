import glob, sys, os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import bct
import matplotlib.pyplot as plt

def readfiles(datafile: str, civpath: str, civL: str, civR: str, measure: str) -> pd.DataFrame:
    data = pd.read_csv(datafile, dtype=str, index_col=['eid'])

    # Read civet files matching data
    Lfiles = [glob.glob(civpath + '*' + str(i) + '*' + civL) for i,row in data.iterrows()]
    Lfiles.sort()
    Lfiles = [item for sublist in Lfiles for item in sublist]
    Rfiles = [glob.glob(civpath + '*' + str(i) + '*' + civR) for i,row in data.iterrows()]
    Rfiles.sort()
    Rfiles = [item for sublist in Rfiles for item in sublist]

    # Read text files
    left_dfs = [pd.read_csv(f, header=None).T for f in Lfiles]
    leftdata = pd.concat(left_dfs,ignore_index=True)
    right_dfs = [pd.read_csv(f, header=None).T for f in Rfiles]
    rightdata = pd.concat(right_dfs,ignore_index=True)

    # Create dataframe
    df=pd.concat([leftdata,rightdata], axis=1)
    df.columns=["V"+str(i) for i in range(1, df.shape[1] + 1)]

    df.index=data.index.copy()
    mean_anat = pd.DataFrame(df.mean(axis=1), columns=["mean_" + measure], index=df.index.copy())
    data = data.join(mean_anat)
    total_anat = pd.DataFrame(df.sum(axis=1), columns=["total_" + measure], index=df.index.copy())
    data = data.join(total_anat)
    df.to_pickle(measure + '.pkl')
    return data, df

def parcellate(DKTfile: str, outdir: str, data: pd.DataFrame, df = pd.DataFrame) -> pd.DataFrame:
    dkt = pd.read_csv(DKTfile, header=None)
    dkt.columns = ['roi']
    rois = dkt.roi.unique()
    rois = np.sort(rois)
    parc = pd.DataFrame(index=data.index.copy())

    for r in rois:
        means = pd.DataFrame(df.iloc[:,dkt.index[dkt.roi==r].tolist()].mean(axis=1),columns=["DKT_"+str(r)], index=data.index.copy())
        parc = pd.concat([parc,means], axis = 1)
    data_parc = pd.concat([data,parc], axis = 1)
    data_parc = data_parc.drop(['DKT_6', 'DKT_106'], axis=1)
    parc = parc.drop(['DKT_6', 'DKT_106'], axis=1)

    parc.to_csv(outdir + 'dkt_parcellation.csv') # parcellated data
    data_parc.to_csv(outdir + 'data_dkt_parcellation.csv') # parcellated data combined with behavioural data
    return parc, data_parc

def bireg(measure: str, parc: pd.DataFrame, data_parc: pd.DataFrame, outdir: str) -> pd.DataFrame:
    data_parc['mean_' + measure] = pd.to_numeric(data_parc['mean_' + measure])
    for r in parc.columns:
        data_parc[r] = pd.to_numeric(data_parc[r])
    # SPLIT SAMPLE INTO TRAIN AND TEST
    train = data_parc.sample(frac=0.8)
    test = data_parc.drop(train.index)
    train.to_csv('SCoRe_train_data.csv')
    test.to_csv('SCoRe_test_data.csv')

    params = pd.DataFrame()
    resids = pd.DataFrame(index=train.index.copy())
    conn_train = np.zeros((len(parc.columns), len(parc.columns), len(train)))
    conn_test = np.zeros((len(parc.columns), len(parc.columns), len(test)))

    # Loop through ROIs and run regressions
    for i in range(len(parc.columns)):
        r1 = parc.columns[i]
        for j in range(len(parc.columns)):
            r2 = parc.columns[j]
            if r1 != r2:
                train_res = sm.ols("train[r1] ~ train[r2] + mean_" + measure, data = train).fit()
                params[r1 + '_' + r2] = train_res.params
                resids[r1 + '_' + r2] = train_res.resid
                test_res = train_res.params[0] + train_res.params[1]*test[r2] + train_res.params[2]*test['mean_' + measure]
                test_residual = test[r1] - test_res
                for p in range(len(train)):
                    conn_train[i,j,p] = train_res.resid.iloc[p]
                for pp in range(len(test)):
                    conn_test[i,j,pp] = test_residual.iloc[pp]

    # Save residuals and regression parameters
    params.to_csv(outdir + 'SCoRe_regression_parameters.csv')
    np.save(outdir + 'SCoRe_3Dconnectivity_matrix_train.npy', conn_train)
    np.save(outdir + 'SCoRe_3Dconnectivity_matrix_test.npy', conn_test)
    merged_data = pd.concat([train, test])
    merged_conn = np.concatenate([conn_train, conn_test], -1)
    return merged_data, merged_conn

def graphmes(conn: pd.DataFrame, data: pd.DataFrame, outdir: str) -> pd.DataFrame:
    strengths = np.zeros((np.size(conn,2),np.size(conn,0)))
    efficiency = np.zeros((np.size(conn,2),1))

    for p in range(np.size(conn, 2)):
        W = conn[:,:,p]
        W = np.abs(W)
        st = bct.strengths_dir(W)
        efficiency[p] = bct.efficiency_wei(W, local=False)
        for r in range(np.size(st)):
            strengths[p,r] = st[r]

    eff = [item for sublist in efficiency for item in sublist]
    e = pd.DataFrame(data = eff, columns = ['Global_Efficiency'], index = data.index.copy())
    np.size(strengths,1)
    s = pd.DataFrame(data = strengths, index = data.index.copy())
    s.columns=["Strength_"+str(i) for i in range(1, s.shape[1] + 1)]
    data_parc_conn = data.join(e)
    data_parc_conn = data_parc_conn.join(s)    
    data_parc_conn.to_csv(outdir + 'score_output.csv')

if __name__ == "__main__":
    measure = "thickness"
    datafile = '/project/def-mlepage/UKBB/current_civet.csv'
    civpath = "/project/def-mlepage/UKBB/civet/thickness/"
    civL = "*native_rms_rsl_tlaplace_20mm_left.txt"
    civR = "*native_rms_rsl_tlaplace_20mm_right.txt"
    DKTfile = "/project/def-mlepage/UKBB/civet/CIVET_2.0_DKT.txt"
    outdir = "/scratch/katie/score/"

    data, df = readfiles(datafile, civpath, civL, civR, measure)
    parc, data_parc = parcellate(DKTfile, outdir, data, df)
    #merged_data, merged_conn = bireg(measure, parc, data_parc, outdir)
    #graphmes(merged_conn, merged_data, outdir)
