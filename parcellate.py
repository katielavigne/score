import pandas as pd

df = pd.read_pickle('/scratch/katie/score/ukbb_vertexdata.pkl')
dktvert = pd.read_csv('/project/def-mlepage/UKBB/civet/CIVET_2.0_DKT.txt', dtype=str, names=['roi'], header=None)
dktinfo = pd.read_csv('/project/def-mlepage/UKBB/civet/DKT.csv', dtype=str)

parc = pd.DataFrame(index=df.index.copy())
for r in range(len(dktinfo)):
    roi = dktinfo.label_number[r]
    abr = dktinfo.abbreviation[r]
    means = pd.DataFrame(df.iloc[:,dktvert.index[dktvert.roi == roi]].mean(axis=1),columns=[abr], index=df.index.copy())
    parc = pd.concat([parc,means], axis = 1)

parc.to_csv('/scratch/katie/score/dkt_parcellation.csv') # parcellated data