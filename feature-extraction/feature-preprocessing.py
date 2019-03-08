import numpy as np

my_data = np.genfromtxt('dataset.csv', delimiter = ',', dtype=None)
my_data_final = my_data[:1001]
header = np.array([['Date', 'Open', 'High', 'Low', 'Close', 'Total Trade Quantity']])
for i in range(2, 11):
    shifted_data_set = my_data[i:1000 + i]
    shifted_data_set = np.concatenate((header, shifted_data_set), axis = 0)
    my_data_final = np.concatenate((my_data_final, shifted_data_set), axis= 1)
my_data_final = my_data_final[:, :59]
my_data_final = np.delete(my_data_final, [56, 57], axis = 1)
np.savetxt("features-with-date.csv", my_data_final, delimiter=',', fmt='%s')

my_data_final2 = np.delete(my_data_final, [_ for _ in range(0, 56, 6)], axis = 1)
my_data_final2 = np.delete(my_data_final2, 0, axis = 0)
my_data_final2 = my_data_final2.astype(float)
(my_data_final2.T)[-1] = (my_data_final2.T)[-1] // (my_data_final2.T)[-2]  
np.savetxt("features-without-date.csv", my_data_final2, delimiter=',', fmt='%f')

