import newton_method
import numpy

data_0 = ''
data_1 = ''

with open(f'../data/data-1.txt', 'r') as data:
    line = data.readline()
    inited_0 = False
    inited_1 = False
    while line:
        line = line.split()
        if (line[len(line)-1]=='0'):
            line.pop()
            if (not inited_0):
                data_0 = numpy.reshape(numpy.array(line), (1, -1)).astype(numpy.float)
                print('inited_0', line)
                inited_0 = True
            else:
                data_0 = numpy.append(data_0, numpy.array(line).reshape(1, -1).astype(numpy.float), 0)
        if ((line[len(line)-1]=='1')):        
            line.pop()
            if (not inited_1):
                data_1 = numpy.reshape(numpy.array(line), (1, -1)).astype(numpy.float)
                print('inited_1', line)
                inited_1 = True
            else:
                data_1 = numpy.append(data_1, numpy.array(line).reshape(1, -1).astype(numpy.float), 0)
        
        line = data.readline()

print(data_0.shape)
print(data_1[:42, :].shape)

newton_method.logistic_regression(data_0, data_1[:42, :], size=42, dim=6)