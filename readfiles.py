import os, glob, pandas as pd

civpath = "/project/def-mlepage/UKBB/civet/thickness/"
Lfiles = glob.glob(civpath + '*left*')
Lfiles.sort()
Rfiles = glob.glob(civpath + '*right*')
Rfiles.sort()
subjIDs = [ids.split('/')[-1].split('_')[1] for ids in Lfiles]

Ldf = pd.concat((pd.read_csv(Lf, dtype=float, header=None).T for Lf in Lfiles))
Ldf.to_pickle('ukbb_leftdata.pkl')
Rdf = pd.concat((pd.read_csv(Rf, dtype=float, header=None).T for Rf in Rfiles))
Rdf.to_pickle('ukbb_rightdata.pkl')
df = pd.concat([Ldf,Rdf], axis=1)
df.index = subjIDs
df.to_pickle('ukbb_vertexdata.pkl')