import pickle
import mindspore
from mindspore import Tensor

pickle_path_numpy = '/mnt/e/hjl/LOGO/file_for_logo/video_feature_dict_numpy.pkl'
pickle_path_ms = '/mnt/e/hjl/LOGO/file_for_logo/video_feature_dict_ms.pkl'

pickle_data_ms = dict()

with open(pickle_path_numpy, 'rb') as f:
    pickle_data_numpy = pickle.load(f)
    for key, value in pickle_data_numpy.items():
        pickle_data_numpy[key] = Tensor.from_numpy(value)
        print(pickle_data_numpy[key])

'''with open(pickle_path_ms, 'wb') as f:
    pickle.dump(pickle_data_ms, f)'''

'''with open(pickle_path_ms, 'rb') as f:
    pickle_data_ms = pickle.load(f)
    print(len(pickle_data_ms))
    for key, value in pickle_data_ms.items():
        print(value.asnumpy())'''