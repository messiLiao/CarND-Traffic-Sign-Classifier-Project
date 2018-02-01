try:
    import pickle
except:
    import cPickle as pickle 

fn = './dataset/traffic-signs-data/valid.p'
with open(fn, 'rb') as fd:
    data = pickle.load(fd)
    wfd = open('./dataset/traffic-signs-data/valid.2.p', 'wb')
    pickle.dump(data, wfd, protocol=2)
    wfd.close()
fn = './dataset/traffic-signs-data/train.p'
with open(fn, 'rb') as fd:
    data = pickle.load(fd)
    wfd = open('./dataset/traffic-signs-data/train.2.p', 'wb')
    pickle.dump(data, wfd, protocol=2)
    wfd.close()
fn = './dataset/traffic-signs-data/test.p'
with open(fn, 'rb') as fd:
    data = pickle.load(fd)
    wfd = open('./dataset/traffic-signs-data/test.2.p', 'wb')
    pickle.dump(data, wfd, protocol=2)
    wfd.close()