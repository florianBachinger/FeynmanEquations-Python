import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import Feynman.Functions as ff

# f: q2*Ef

########## without noise ##########
df = ff.Feynman12_data()

###### plotting ######

#pairplot to see distribution
g = sns.pairplot(df)
plt.savefig('fig/pairplot.png')

plt.clf()

#linechart so see prediction
df = df.sort_values(by=['q2', 'Ef',])
x = range(0,len(df))
for col in list(df.columns):
  sns.lineplot(data = df, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot1.png')

plt.clf()

#linechart so see prediction
df = df.sort_values(by=[ 'Ef', 'q2',])
x = range(0,len(df))
for col in list(df.columns):
  sns.lineplot(data = df, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot2.png')

plt.clf

##########   with noise   ##########
df = ff.Feynman12_data(noise_level=0.3)

###### plotting ######

#pairplot to see distribution
g = sns.pairplot(df)
plt.savefig('fig/pairplot_noisy.png')

plt.clf()

#linechart so see prediction
df = df.sort_values(by=['q2', 'Ef',])
x = range(0,len(df))
for col in list(df.columns):
  sns.lineplot(data = df, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot1_noisy.png')

plt.clf()

#linechart so see prediction
df = df.sort_values(by=[ 'Ef', 'q2',])
x = range(0,len(df))
for col in list(df.columns):
  sns.lineplot(data = df, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot2_noisy.png')