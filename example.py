import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Feynman.Functions import Feynman12, FunctionsJson

########## testing ##########
# f: q2*Ef

inputSize = 10000
X = np.random.uniform([1.0,1.0], [5.0,5.0], (inputSize,2))
q2 = X[:,0] 
Ef = X[:,1] 
# f: q2*Ef
f1 = Feynman12.calculate(q2,Ef)

jsonArr = np.array(FunctionsJson)
json = jsonArr[ [ row['EquationName']== 'Feynman12' for row in FunctionsJson ]  ][0]
eq = eval(json['Formula_Lambda'])
f2 = [eq(row) for row in X]

print(f"two variants equal {(f1==f2).all()}")


########## without noise ##########
###### plotting ######
df = Feynman12.generate_df(size = inputSize, noise_level=0.3)

df_noNoise = df[['q2', 'Ef', 'F_without_noise']]
df_noNoise.columns = ['q2', 'Ef', 'F']
#pairplot to see distribution
g = sns.pairplot(df_noNoise)
plt.savefig('fig/pairplot.png')

plt.clf()

#linechart so see prediction
df_noNoise = df_noNoise.sort_values(by=['q2', 'Ef',])
x = range(0,len(df_noNoise))
for col in list(df_noNoise.columns):
  sns.lineplot(data = df_noNoise, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot1.png')

plt.clf()

#linechart so see prediction
df_noNoise = df_noNoise.sort_values(by=[ 'Ef', 'q2',])
x = range(0,len(df_noNoise))
for col in list(df_noNoise.columns):
  sns.lineplot(data = df_noNoise, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot2.png')

plt.clf

##########   with noise   ##########
###### plotting ######

df_noise = df[['q2', 'Ef', 'F']]
#pairplot to see distribution
g = sns.pairplot(df_noise)
plt.savefig('fig/pairplot_noisy.png')

plt.clf()

#linechart so see prediction
df_noise = df_noise.sort_values(by=['q2', 'Ef',])
x = range(0,len(df_noise))
for col in list(df_noise.columns):
  sns.lineplot(data = df_noise, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot1_noisy.png')

plt.clf()

#linechart so see prediction
df_noise = df_noise.sort_values(by=[ 'Ef', 'q2',])
x = range(0,len(df_noise))
for col in list(df_noise.columns):
  sns.lineplot(data = df_noise, x=x, y=col, label = col, alpha=0.7 )
plt.savefig('fig/lineplot2_noisy.png')