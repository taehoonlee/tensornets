import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mistune import html  # pip install mistune==2.0.0a4
from bs4 import BeautifulSoup  # pip install bs4

with open('README.md') as f:
    contents = BeautifulSoup(html(f.read()), 'html.parser')

table = []
for row in contents.select('table')[0].select('tr'):
    if len(row.find_all('td')) > 0:
        table.append([])
        for col in row.find_all('td')[:-3]:
            table[-1].append(col.text)

table = np.array(table)
name = table[:, 0]
top1 = np.asarray(table[:, 2], dtype=np.float)
top5 = np.asarray(table[:, 3], dtype=np.float)
mac = np.asarray([x[:-1] for x in table[:, 4]], dtype=np.float)
size = np.asarray([x[:-1] for x in table[:, 5]], dtype=np.float)

groups = []
groups.append(
    ('ResNet', '#EF5350', np.array([('ResNet' in x) and (len(x) < 10) for x in name])))
groups.append(
    ('ResNet2', '#EC407A', np.array([('ResNet' in x) and ('v2' in x) for x in name])))
groups.append(
    ('ResNeXt', '#AB47BC', np.array([('ResNeXt' in x) and ('c32' in x) for x in name])))
groups.append(
    ('Inception', '#5C6BC0', np.array([('Inception' in x) for x in name])))
groups.append(
    ('DenseNet', '#29B6F6', np.array([('DenseNet' in x) for x in name])))
# groups.append(
#     ('MobileNet', '#26A69A', np.array([('MobileNet' in x) and (len(x) < 13) for x in name])))
groups.append(
    ('MobileNet2', '#66BB6A', np.array([('MobileNet' in x) and ('v2' in x) for x in name])))
groups.append(
    ('MobileNet3', '#9CCC65', np.array([('v3large' in x) and ('mini' not in x) for x in name])))
groups.append(
    ('EfficientNet', '#FFA726', np.array([('EfficientNet' in x) for x in name])))

f, axarr = plt.subplots(2, 2, figsize=(8, 8))
for (label, color, index) in groups:
    kwargs = {'label': label, 'ls': '--', 'linewidth': 1,
              'marker': 'o', 'markersize': 5, 'color': color}
    axarr[0, 0].plot(size[index], top1[index], **kwargs)
    axarr[0, 1].plot(size[index], top5[index], **kwargs)
    axarr[1, 0].plot(mac[index], top1[index], **kwargs)
    axarr[1, 1].plot(mac[index], top5[index], **kwargs)

axarr[0, 0].legend()
axarr[0, 0].set_xlabel('Size (M)')
axarr[0, 0].set_xscale('log')
axarr[0, 0].set_ylabel('Top-1 (%)')

axarr[0, 1].set_xlabel('Size (M)')
axarr[0, 1].set_xscale('log')
axarr[0, 1].set_ylabel('Top-5 (%)')

axarr[1, 0].set_xlabel('MAC (M)')
axarr[1, 0].set_xscale('log')
axarr[1, 0].set_ylabel('Top-1 (%)')

axarr[1, 1].set_xlabel('MAC (M)')
axarr[1, 1].set_xscale('log')
axarr[1, 1].set_ylabel('Top-5 (%)')

for i in range(2):
    for j in range(2):
        axarr[i, j].grid(linestyle=':')
        axarr[i, j].minorticks_on()
        axarr[i, j].tick_params(axis='both', which='both', direction='in')

plt.tight_layout()
plt.savefig('summary.png', dpi=200)
