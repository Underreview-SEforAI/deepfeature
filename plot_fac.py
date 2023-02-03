# seaborn barplot that shows the Feature map Accuracy Coverage~(FAC) of the 12 features
# please plot the fig, the x-axis is the feature map index, and the y-axis is the FAC
# please use the seaborn package to plot the barplot
# you should consider decide the min of the y-axis and the max of the y-axis
# set_ylim(min(FAC_list), max(FAC_list))
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


# svhn lenet5
# a = np.array([58.85064816505386, 35.15610735804272, 3.4507942304181114, 48.26547379952528, 32.90578784005844, 56.687054957093295])
# b = np.array([68.73288296512689, 44.09348183312032, 3.729231331020628, 60.986854117217455, 42.42285922950521, 66.07631915282089])
# c = np.array([68.65985028300165, 44.11174000365163, 3.7064086178564963, 59.29340880043819, 42.74694175643601, 66.93445316779258])
# d = np.array([67.72868358590469, 43.07558882599963, 3.911813036333754, 62.812671170348736, 41.09914186598503, 66.39127259448603])


# cifar10 resnet20
# a = np.array([11.683168316831683, 18.04309842748981, 10.75131042516017, 17.588817705299945, 7.920792079207914, 9.015725101921959, 14.478741991846249, 9.7146185206756, 7.478159580663942, 16.598718695398958, 17.285963890506693, 44.84566103669191, 16.691904484566095, 16.039603960396036, 5.195107746068729, 14.5719277810134])
# b = np.array([15.9231217239371, 24.47291788002329, 13.93127548048922, 23.412929528246934, 9.702970297029694, 11.357018054746646, 18.380896913220738, 14.175888177052997, 8.025626092020971, 30.297029702970306, 23.762376237623755, 66.09202096680256, 22.748980780430983, 20.74548631333721, 5.579499126383226, 22.189866045428076])
# c = np.array([17.029702970297038, 24.962143273150843, 14.012813046010493, 23.77402446126966, 10.425160163075134, 11.333721607454862, 19.266161910308682, 13.476994758299355, 7.897495631916129, 32.591729761211425, 23.59930110658125, 68.28188701223064, 23.04018637157833, 20.559114735002908, 4.577751892836346, 23.145020384391373])
# d = np.array([17.845078625509615, 24.8573092603378, 12.882935352358757, 23.21490972626674, 10.390215492137443, 10.413511939429227, 19.17297612114153, 13.977868375072802, 7.105416423995337, 35.771694816540474, 21.86371578334304, 70.94933022714036, 23.063482818870114, 19.638905066977287, 4.216656959813619, 24.065230052417007])

# cifar10 resnet20 block 2
# a = np.array([34.45544554455445, 34.304018637157824, 44.41467676179383, 48.89924286546302, 21.712288875946413, 23.203261502620848, 22.11997670355271, 31.077460687245193, 21.3744903902155, 32.34711706464765, 27.11706464764123, 43.41292952824695, 27.256843331391963, 42.20151426907397, 31.624927198602222, 20.7804309842749])
# b = np.array([47.75771694816541, 48.5497961560862, 65.36983110075714, 69.05066977285964, 34.117647058823536, 33.02271403610949, 36.69190448456611, 38.439138031450206, 34.89807804309842, 48.153756552125806, 37.76354105998835, 64.4612696563774, 46.26674432149097, 63.76237623762376, 50.23878858474083, 33.86138613861385])
# c = np.array([48.9341875364007, 49.79615608619685, 68.31683168316832, 71.60163075131042, 37.00640652300524, 35.15433896330809, 40.23296447291788, 38.73034362259755, 36.48223645894001, 52.58008153756552, 37.58881770529994, 67.1520093185789, 49.36517181129878, 67.66453115899824, 53.41875364006989, 36.33080955154339])
# d = np.array([50.07571345369831, 52.58008153756552, 70.18054746651136, 73.70995923121724, 37.8683750728014, 35.65521258008154, 44.892253931275484, 41.01339545719278, 38.683750728013976, 55.57367501456028, 37.78683750728013, 68.92253931275481, 51.32207338380897, 70.25043680838672, 55.77169481654048, 39.00990099009901])

# svhn lenet5 layer 2
a = np.array([55.51853204308928, 56.59576410443673, 67.37721380317691, 3.523826912543356, 46.21599415738543, 51.1183129450429, 3.4553587730509463, 68.1668796786562, 45.56326456089101, 63.74840241007851, 44.34909622055871, 52.264013145882785, 3.647069563629728, 54.75168888077415, 46.95088552127077, 60.35694723388716])
b = np.array([64.88953806828556, 66.58754792769764, 76.22786196823078, 3.4325360598868, 56.35840788752967, 61.70348731057148, 3.295599780901952, 75.96768303815958, 57.39455906518167, 71.10644513419756, 52.59722475807924, 64.25963118495527, 3.327551579331754, 65.1725397115209, 57.978820522183675, 71.40770494796422])
c = np.array([64.93974803724667, 66.63775789665875, 76.81212342523278, 3.441665145152456, 57.68212525104984, 62.17363520175279, 3.3321161219645745, 76.79386525470147, 58.20248311119226, 70.47197370823443, 52.227496804820156, 64.346357494979, 3.409713346722654, 66.18130363337593, 58.266386708051854, 71.45791491692532])
d = np.array([65.00365163410626, 65.3642505020997, 76.75734891363885, 3.4873105714807338, 57.68212525104984, 63.09567281358408, 3.6379404783640723, 77.51506299068834, 59.0834398393281, 70.5313127624612, 52.71590286653277, 65.19079788205222, 3.60142413730145, 66.36844988132188, 59.35274785466496, 72.61730874566368])
a_name = ['3-steps'] * len(a)
a = [x*0.01 for x in a]

b_name = ['7-steps'] * len(a)
b = [x*0.01 for x in b]
c_name = ['10-steps'] * len(a)
c = [x*0.01 for x in c]
d_name = ['20-steps'] * len(a)
d = [x*0.01 for x in d]

idx = [i for i in range(1,len(a)+1)] * 4
# print(len(idx))

data = np.concatenate([a,b,c,d], axis=0)
name = np.concatenate([a_name, b_name, c_name, d_name], axis=0)
# print(name)

plt.figure(figsize=(8,4))
fig, ax = plt.subplots()
table = pd.DataFrame({'FAS value (Attack Success Rate)':data, 'num_steps':name, 'idx':idx})
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
sns.set_theme(style='whitegrid')
# plt.ylim((min(a), max(d)))
# plt.xticks(range(1,33,2))
# sns.dark_palette(sns.color_palette("hls")[-5])
# sns.cubehelix_palette(as_cmap=True)
sns.barplot(data = table, x = 'idx', y='FAS value (Attack Success Rate)', hue='num_steps', palette="Greens")
plt.xticks(range(0,len(a)+1))
plt.savefig('./deepfeature/rq1/FAC_svhn_lenet5_layer2.pdf', bbox_inches='tight')