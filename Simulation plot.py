import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_absolute_percentage_error

plt.style.use('fivethirtyeight')
train_range=672
def animate1(i):
    data = pd.read_csv('dat.csv')
    x = data['Unnamed: 0']
    x+=(train_range+1)
    y1 = data['Actual']
    z1 = data['CNN-LSTM']
    z2 = data['CNN']
    z3= data['LSTM']


    if (len(x.index)+train_range)<=192+train_range:
        plt.cla()
        #plot 1
        # plt.subplot(1,3,1)
        plt.title("Simulasi")
        plt.plot(x, y1, '--', label='Actual, total data : %d' % (len(x.index)+train_range))
        plt.plot(x, z1, label='CNN-LSTM, MAPE : %.3f Data: %d' % ((mean_absolute_percentage_error(y1[0:len(x.index)], z1[0:len(x.index)])),len(z1.index)+train_range))
        plt.plot(x, z2, label='CNN, MAPE : %.3f Data: %d' % ((mean_absolute_percentage_error(y1[0:len(x.index)], z2[0:len(x.index)])),len(z2.index)+train_range))
        plt.plot(x, z3, label='LSTM, MAPE : %.3f Data: %d' % ((mean_absolute_percentage_error(y1[0:len(x.index)], z3[0:len(x.index)])),len(z3.index)+train_range))
        plt.xlim(train_range,(len(x.index)+train_range))
        plt.ylim(0,max(y1)+200)
        plt.xlabel('Timestep')
        plt.ylabel('Output Volume')
        plt.legend(loc='upper left')
        # #plot 2
        # plt.subplot(1,3,2)
        # plt.title("42-2")
        # plt.plot(x, y2, label='42-1')
        # plt.plot(x, z2, label='Forecast 42-1')
        # plt.xlim(0,len(x.index))
        # plt.legend(loc='upper left')
        # #plot 3
        # plt.subplot(1,3,3)
        # plt.title("42-3")
        # plt.plot(x, y3, label='42-1')
        # plt.plot(x, z3, label='Forecast 42-1')
        # plt.xlim(0,len(x.index))
        # plt.legend(loc='upper left')
        plt.tight_layout()
    else:
        plt.cla()
        #plot 1
        # plt.subplot(1,3,1)
        plt.title("Simulasi")
        plt.plot(x, y1, '--', label='Actual, total data : %d' % (len(x.index)+train_range))
        plt.plot(x, z1, label='CNN-LSTM, MAPE : %.3f Data: %d' % ((mean_absolute_percentage_error(y1[len(x.index)-96:len(x.index)], z1[len(x.index)-96:len(x.index)])),len(z1.index)+train_range))
        plt.plot(x, z2, label='CNN, MAPE : %.3f Data: %d' % ((mean_absolute_percentage_error(y1[len(x.index)-96:len(x.index)], z2[len(x.index)-96:len(x.index)])),len(z2.index)+train_range))
        plt.plot(x, z3, label='LSTM, MAPE : %.3f Data: %d' % ((mean_absolute_percentage_error(y1[len(x.index)-96:len(x.index)], z3[len(x.index)-96:len(x.index)])),len(z3.index)+train_range))
        plt.xlim((len(x.index)+train_range)-192,(len(x.index)+train_range))
        plt.ylim(0,max(y1)+200)
        plt.xlabel('Timestep')
        plt.ylabel('Output Volume')
        plt.legend(loc='upper left')
        # #plot 2
        # plt.subplot(1,3,2)
        # plt.title("42-2")
        # plt.plot(x, y2, label='42-1')
        # plt.plot(x, z2, label='Forecast 42-1')
        # plt.xlim(len(x.index)-1000,len(x.index))
        # plt.legend(loc='upper left')
        # #plot 3
        # plt.subplot(1,3,3)
        # plt.title("42-3")
        # plt.plot(x, y3, label='42-1')
        # plt.plot(x, z3, label='Forecast 42-1')
        # plt.xlim(len(x.index)-1000,len(x.index))
        # plt.legend(loc='upper left')
        plt.tight_layout()
ani1 = FuncAnimation(plt.gcf(), animate1, interval=1000)
plt.tight_layout()
plt.show()