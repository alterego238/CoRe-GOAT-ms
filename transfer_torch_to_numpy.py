import pickle
import torch


'''pickle_path_torch = '/mnt/disk_1/jiale_intern/file_for_logo/video_feature_dict.pkl'
pickle_path_numpy = '/mnt/disk_1/jiale_intern/file_for_logo/video_feature_dict_numpy.pkl'
'''

pickle_path_torch = '/mnt/disk_1/jiale_intern/file_for_logo/swin_features_dict_new.pkl'
pickle_path_numpy = '/mnt/disk_1/jiale_intern/file_for_logo/swin_features_dict_new_numpy.pkl'


'''pickle_path_torch = '/mnt/disk_1/jiale_intern/file_for_logo/formation_features_middle_1.pkl'
pickle_path_numpy = '/mnt/disk_1/jiale_intern/file_for_logo/formation_features_middle_1_numpy.pkl'
'''

pickle_data_numpy = dict()

with open(pickle_path_torch, 'rb') as f:
    pickle_data_torch = pickle.load(f)
    for key, value in pickle_data_torch.items():
        pickle_data_numpy[key] = value.numpy()

with open(pickle_path_numpy, 'wb') as f:
    pickle.dump(pickle_data_numpy, f)