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

def apply_score(measure: str, paramfile: str, parc: pd.DataFrame, data_parc: pd.DataFrame, outdir: str) -> pd.DataFrame:
    data_parc['mean_' + measure] = pd.to_numeric(data_parc['mean_' + measure])
    for r in parc.columns:
        data_parc[r] = pd.to_numeric(data_parc[r])
    params = pd.read_csv(paramfile, index_col=0)
    resids = pd.DataFrame(index=data_parc.index.copy())
    conn = np.zeros((len(parc.columns), len(parc.columns), len(data_parc)))

    # Loop through ROIs and apply regression parameters
    for i in range(len(parc.columns)):
        r1 = parc.columns[i]
        for j in range(len(parc.columns)):
            r2 = parc.columns[j]
            if r1 != r2:
                reg = params.loc['Intercept', r1 + '_' + r2] + params.loc['train[r2]', r1 + '_' + r2]*data_parc[r2] + params.loc['mean_thickness', r1 + '_' + r2]*data_parc['mean_' + measure]
                residual = parc[r1] - reg
                for pp in range(len(data_parc)):
                    conn[i,j,pp] = residual.iloc[pp]

    # Save residuals and regression parameters
    np.save(outdir + 'SCoRe_3Dconnectivity_matrix.npy', conn)
    return conn

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
    data_parc_conn.to_csv(outdir + 'apply_score_output.csv')

if __name__ == "__main__":
    measure = "thickness"
    datafile = '/scratch/katie/ukbb/Katie_2022-03-23.csv'
    civpath = "/project/def-mlepage/UKBB/civet/thickness/"
    civL = "*native_rms_rsl_tlaplace_20mm_left.txt"
    civR = "*native_rms_rsl_tlaplace_20mm_right.txt"
    DKTfile = "/project/def-mlepage/UKBB/civet/CIVET_2.0_DKT.txt"
    outdir = "/scratch/katie/score/"
    paramfile = "/scratch/katie/score/SCoRe_regression_parameters.csv"

    data, df = readfiles(datafile, civpath, civL, civR, measure)
    parc, data_parc = parcellate(DKTfile, outdir, data, df)
    conn = apply_score(measure, paramfile, parc, data_parc, outdir)
    graphmes(conn, data_parc, outdir)
