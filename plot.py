from matplotlib import pyplot as plt

def plot1():
    x_list = range(4)
    y_list = [10, 90, 99, 54, 99, 54]

    plt.hist(y_list)
    plt.xlabel('xlabel')
    plt.ylabel('y')
    plt.show()

def plot_accuracy():

    fn = './result/_acc_03.list'
    fd = open(fn, 'r')
    lines = fd.readlines()
    val1_list = []
    for line in lines:
        if line.find('Validation') >= 0:
            line = line.replace('Validation Accuracy = ', '').replace('\n', '').replace('\r', '')
            print line
            val1_list.append(float(line))

    fn = './result/_acc_02.list'
    fd = open(fn, 'r')
    lines = fd.readlines()
    val2_list = []
    for line in lines:
        if line.find('Validation') >= 0:
            line = line.replace('Validation Accuracy = ', '').replace('\n', '').replace('\r', '')
            print line
            val2_list.append(float(line))

    fn = './result/_acc_04.list'
    fd = open(fn, 'r')
    lines = fd.readlines()
    val3_list = []
    for line in lines:
        if line.find('Validation') >= 0:
            line = line.replace('Validation Accuracy = ', '').replace('\n', '').replace('\r', '')
            print line
            val3_list.append(float(line))
    plt.figure()
    x1, = plt.plot(val1_list[:200])
    x2, = plt.plot(val2_list[:200])
    x3, = plt.plot(val3_list[:200])
    plt.legend(handles=[x3, x1, x2], labels=['BATCH_SIZE=64', 'BATCH_SIZE=128', 'BATCH_SIZE=512'], loc='best')
    plt.title('Validation Accuracy / BATCH_SIZE=64/128/512')
    plt.xlabel('epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig('./examples/ValidationAccuracy_bs_64_128_512.png')
    plt.show()

    pass

plot_accuracy()