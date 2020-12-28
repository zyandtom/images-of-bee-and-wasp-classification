import numpy as np
import  matplotlib.pyplot as plt

xloss = np.array([2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000])

yloss = np.array([[2.227, 1.887, 1.698, 1.587, 1.522, 1.465, 1.384, 1.348, 1.316, 1.290, 1.306, 1.264],#initial
                  [2.303, 2.294, 2.100, 1.844, 1.707, 1.626, 1.550, 1.489, 1.445, 1.425, 1.393, 1.370],#add 1 conv
                  [2.287, 2.039, 1.871, 1.748, 1.640, 1.582, 1.506, 1.468, 1.433, 1.407, 1.394, 1.347],#add 1 fulc
                  [2.303, 2.250, 2.000, 1.835, 1.710, 1.652, 1.569, 1.535, 1.457, 1.437, 1.413, 1.394],#add 2 fulc
                  [2.191, 1.806, 1.595, 1.512, 1.426, 1.392, 1.306, 1.280, 1.235, 1.227, 1.196, 1.161],#add filters
                  [2.272, 1.924, 1.745, 1.672, 1.603, 1.561, 1.512, 1.503, 1.467, 1.439, 1.447, 1.405],#reduce filters
                  [2.304, 2.301, 2.294, 2.274, 2.231, 2.180, 2.121, 2.085, 2.029, 1.971, 1.896, 1.887],#0.0001 lr
                  [2.074, 1.956, 1.918, 1.937, 1.944, 1.961, 1.970, 1.981, 1.988, 1.981, 1.966, 1.959],#0.01 lr
                  [1.921, 1.654, 1.548, 1.474, 1.441, 1.383, 1.320, 1.292, 1.299, 1.244, 1.275, 1.260],#Adam
                  [2.100, 1.984, 1.941, 1.911, 1.885, 1.872, 1.839, 1.823, 1.802, 1.800, 1.806, 1.782],#Adagrad
                  [1.860, 1.620, 1.499, 1.453, 1.420, 1.389, 1.313, 1.316, 1.308, 1.268, 1.251, 1.264],#RMSprop
                  [2.303, 2.211, 2.019, 1.900, 1.782, 1.687, 1.591, 1.546, 1.542, 1.496, 1.468, 1.457],#without norm
                  [2.133, 1.780, 1.658, 1.623, 1.544, 1.501, 1.434, 1.386, 1.355, 1.334, 1.330, 1.315] #with norm
                  ])

p1 = plt.plot(xloss,yloss[11],'r--',label='without normalization accuracy 0.49')
#p2 = plt.plot(xloss,yloss[1],'g--',label='add 1 more conv layer with accuracy 0.51')
#p3 = plt.plot(xloss,yloss[2],'b--',label='add 1 more fulc layer with accuracy 0.57')
#p4 = plt.plot(xloss,yloss[3],'y--',label='add 2 more fulc layer with accuracy 0.50')
#p5 = plt.plot(xloss,yloss[4],'g--',label='add filters with accuracy 0.59')
#p6 = plt.plot(xloss,yloss[5],'b--',label='reduce filters with accuracy 0.47')
#p7 = plt.plot(xloss,yloss[6],'g--',label='lr = 0.0001 with accuracy 0.33')
#p8 = plt.plot(xloss,yloss[7],'b--',label='lr = 0.01 with accuracy 0.21')
#p9 = plt.plot(xloss,yloss[8],'g--',label='Adam optimizer with accuracy 0.55')
#p10 = plt.plot(xloss,yloss[9],'b--',label='Adagrad optimizer with accuracy 0.36')
#p11 = plt.plot(xloss,yloss[10],'y--',label='RMSprop optimizer with accuracy 0.52')
p12 = plt.plot(xloss,yloss[12],'g--',label='with normalization accuracy 0.53')


plt.xlabel('mini-batches')
plt.ylabel('loss')
plt.legend()
plt.show()
