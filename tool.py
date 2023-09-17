from os.path import join, exists, dirname, abspath
import numpy as np
import random, os, sys, math
import laspy
from helper_ply import read_ply, write_ply
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

    
class ConfigSensatUrban:
    data_set_dir = '/nas2/jacob/SensatUrban_Data' #data set path
    log_file_dir = 'train_log/Sensat' #log txt file saving path
    train_sum_dir = 'tf_events/Sensat/lovas' #tf events saving path
    saving_path = 'trained_weights/Sensat/lovas' #trained weights saving path
    saving = True
    use_custom_weights = False #using custom class weights. If it is false, it will read class counts from data.
    
    
    log_file_name = 'lovas'
    show_base_only = False #showing performance on base receptive field only
    
    k_n = [16, 21, 21, 21, 16]  # KNN
    num_layers = 5  # Number of layers
    num_points = 28672 # Number of input points
    num_classes = 13  # Number of valid classes
    sub_grid_size = 0.2 # preprocess_parameter

    batch_size = 16 # batch_size during training
    val_batch_size = 4  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 10  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    connection_ratio = 4
    d_out = [16, 64, 128, 256, 512]  # feature dimension
    n_point_connection = [4096, 1024, 256, 64, 16] #Number of Connection Points for each layers

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 5e-3  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    
    loss_function = 'lovas' # wce, lovas, cbloss
    
    use_val_data = False #using validation data for training
    
    save_preset_epoch= False
    preset_1 = 41
    preset_2 = 50
    
    data_augmentation = False
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8

    
class ConfigDALES:
    data_set_dir = '/nas2/jacob/DALES' #data set path
    log_file_dir = 'train_log/DALES' #log txt file saving path
    train_sum_dir = 'tf_events/DALES/SA_CN0_lovas_data_aug_2' #tf events saving path
    saving_path = 'trained_weights/DALES/SA_CN0_lovas_data_aug_2' #trained weights saving path
    saving = True
    use_custom_weights = False #using custom class weights. If it is false, it will read class counts from data.
    
    
    log_file_name = 'SA_CN0_lovas_data_aug_2'
    show_base_only = False #showing performance on base receptive field only
    
    k_n = [16, 21, 21, 21, 16]  # KNN
    num_layers = 5  # Number of layers
    num_points = 28672 #50176#114688#28672#57344 # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.20 # preprocess_parameter

    batch_size = 4 #8#8 # batch_size during training
    val_batch_size = 16 #32#32  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 10  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    connection_ratio = 4
    d_out = [16, 64, 128, 256, 512]  # feature dimension
    n_point_connection = [4096, 1024, 256, 64, 32]#[7168, 1792, 448, 112, 28]#[16384, 4096, 1024, 256, 128]#[4096, 1024, 256, 64, 32]#[8192, 2048, 512, 128, 64] #Number of Connection Points for each layers

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 5e-3  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    
    loss_function = 'lovas' # wce, lovas, cbloss
    
    use_val_data = False #using validation data for training
    
    save_preset_epoch= False
    preset_1 = 41
    preset_2 = 50
    
    data_augmentation = False
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 1.0

    
class ConfigToronto3D:
    data_set_dir = '/nas2/jacob/data/Toronto3D' #data set path
    log_file_dir = 'train_log/Toronto3D' #log txt file saving path
    train_sum_dir = 'tf_events/Toronto3D/lovas_0' #tf events saving path
    saving_path = 'trained_weights/Toronto3D/lovas_0' #trained weights saving path
    saving = True
    use_custom_weights = False #using custom class weights. If it is false, it will read class counts from data.
    
    
    log_file_name = 'lovas_0'
    show_base_only = False #showing performance on base receptive field only
    
    use_rgb = False # Use RGB
    use_intensity = True # Use intensity
    
    k_n = [16, 21, 21, 21, 16]  # KNN
    num_layers = 5  # Number of layers
    num_points = 28672 # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.06 # preprocess_parameter
    
    loss_function = 'lovas'

    batch_size = 16 # batch_size during training
    val_batch_size = 2  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 10  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    connection_ratio = 4
    d_out = [16, 64, 128, 256, 512]  # feature dimension
    n_point_connection = [4096, 1024, 256, 64, 16] #Number of Connection Points for each layers

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 5e-3  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    
    use_val_data = False #using validation data for training
    
    save_preset_epoch= False
    preset_1 = 41
    preset_2 = 50
    
    data_augmentation = False
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001


class DataProcessing:

    @staticmethod
    def get_num_class_from_label(labels, total_class):
        num_pts_per_class = np.zeros(total_class, dtype=np.int32)
        # original class distribution
        val_list, counts = np.unique(labels, return_counts=True)
        for idx, val in enumerate(val_list):
            num_pts_per_class[val] += counts[idx]
        # for idx, nums in enumerate(num_pts_per_class):
        #     print(idx, ':', nums)
        return num_pts_per_class

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    def data_aug_xyz(xyz, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def read_ply_data(path, with_rgb=True, with_label=True, with_i = True):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']
            return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        elif not with_rgb and with_label:
            labels = data['class']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)
        
    @staticmethod
    def read_ply_YU(path):
        data = read_ply(path)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        labels = data['class']
        feature = np.vstack((data['intensity'], data['n_return'])).T
        return xyz.astype(np.float32), feature.astype(np.float32), labels.astype(np.uint8)
        
    @staticmethod
    def read_ply_data_toronto_3D(path, with_rgb=True, with_label=True):
        """
        ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), 
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), 
        ('scalar_Intensity', '<f4'), ('scalar_GPSTime', '<f4'), ('scalar_ScanAngleRank', '<f4'), 
        ('scalar_Label', '<f4')])
        UTM_OFFSET = [627285, 4841948, 0]
        """
        data = read_ply(path)
        xyz = np.vstack((data['x']-627285, data['y']-4841948, data['z'])).T
        if with_rgb and with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['scalar_Label']
            return xyz.astype(np.float32), rgb.astype(np.uint8), labels.astype(np.uint8)
        elif with_rgb and not with_label:
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        elif not with_rgb and with_label:
            labels = data['scalar_Label']
            return xyz.astype(np.float32), labels.astype(np.uint8)
        elif not with_rgb and not with_label:
            return xyz.astype(np.float32)
        
    @staticmethod
    def read_las_data(data_label_filename):
        '''
        loading las file
        ---------------------------------------
        input: las file name
        output: np array with dimension of [num of points, labels]
                labels are [x,y,z, intensity, intensity, classification]
        '''
        inFile = laspy.file.File(data_label_filename)
        xyz = np.vstack((inFile.x, inFile.y, inFile.z)).T
        feature = np.vstack((inFile.get_intensity()/65535, inFile.get_num_returns())).T
        labels = inFile.get_classification()

        return xyz.astype(np.float32), feature.astype(np.float32), labels.astype(np.uint8)

    @staticmethod
    def read_las_no_label(data_label_filename):
        '''
        loading las file
        ---------------------------------------
        input: las file name
        output: np array with dimension of [num of points, labels]
                labels are [x,y,z, intensity, intensity, classification]
        '''
        inFile = laspy.file.File(data_label_filename)
        xyz = np.vstack((inFile.x, inFile.y, inFile.z)).T
        feature = np.vstack((inFile.get_intensity()/65535, inFile.get_num_returns())).T

        return xyz.astype(np.float32), feature.astype(np.float32)
    
    @staticmethod
    def read_las_data_no_norm(data_label_filename):
        '''
        loading las file
        ---------------------------------------
        input: las file name
        output: np array with dimension of [num of points, labels]
                labels are [x,y,z, intensity, intensity, classification]
        '''
        inFile = laspy.file.File(data_label_filename)
        xyz = np.vstack((inFile.x, inFile.y, inFile.z)).T
        feature = np.vstack((inFile.get_intensity()/65535, inFile.get_num_returns())).T
        labels = inFile.get_classification()

        return xyz.astype(np.float32), feature.astype(np.float32), labels.astype(np.uint8)
    
    @staticmethod
    def read_las_no_label_no_norm(data_label_filename):
        '''
        loading las file
        ---------------------------------------
        input: las file name
        output: np array with dimension of [num of points, labels]
                labels are [x,y,z, intensity, intensity, classification]
        '''
        inFile = laspy.file.File(data_label_filename)
        xyz = np.vstack((inFile.x, inFile.y, inFile.z)).T
        feature = np.vstack((inFile.get_intensity()/65535, inFile.get_num_returns())).T

        return xyz.astype(np.float32), feature.astype(np.float32)
    
    @staticmethod
    def random_sub_sampling(points, features=None, labels=None, sub_ratio=10, verbose=0):
        num_input = np.shape(points)[0]
        num_output = num_input // sub_ratio
        idx = np.random.choice(num_input, num_output)
        if (features is None) and (labels is None):
            return points[idx]
        elif labels is None:
            return points[idx], features[idx]
        elif features is None:
            return points[idx], labels[idx]
        else:
            return points[idx], features[idx], labels[idx]

    @staticmethod
    def get_class_weights(num_per_class, name='sqrt', num_class=13 , custom_weights = False, beta=0.9999):
        # # pre-calculate the number of points in each category
        if custom_weights:
            num_per_class = np.array([10,10,10,1,1,1], dtype=np.int32)
        
        frequency = num_per_class / float(sum(num_per_class))
        if name == 'sqrt' or name == 'lovas':
            ce_label_weight = 1 / np.sqrt(frequency)
        elif name == 'wce':
            ce_label_weight = 1 / (frequency + 0.02)
        elif name == 'cbloss':
            effective_num = 1.0 - np.power(beta, num_per_class)
            weights = (1.0 - beta) / np.array(effective_num)
            ce_label_weight = weights / np.sum(weights) * num_class
            
        else:
            raise ValueError('Only support sqrt, wce, and cb_loss')
        return np.expand_dims(ce_label_weight, axis=0)
    
def feature_extraction(voxels:dict, pts_cloud:np.ndarray) -> np.ndarray:
    """This function convert a given voxels grid to a featurized matrix.
    Features names in the feature_names list will be computed for each voxel.

    Args:
        voxels (dict): [description]
        pts_cloud (np.ndarray): [description]
        feature_names (list, optional): [description]. Defaults to None.

    Returns:
        np.ndarray: [description]
    """

    features = []

    for i, (key, pointsIdx) in enumerate(voxels.items()):
        x, y, z, _ = key
        features.append([x,y,z])
        points=pts_cloud[pointsIdx, :]
        feature = points[:, 2].min()
        features[i].append(feature)
    features = np.array(features)
    return features


def devoxelize(predictions:None, voxels:dict, pts_cloud:np.ndarray, maxZ) -> None:
    """
    Post process the network output, and eventually map the label the points cloud.
    Args:
    features (None): [description]
    config (dict): [description]
    """
    print('Calculating elevation...')
    #pbar0 = tqdm(total=predictions.shape[0])

    i=0
    for row in predictions:
        x,y,z = row[[0,1,2]]
        localminZ = row[-1]

        key = (x, y, z, 0)
        pointsIdx = voxels.get(key, None)
        if pointsIdx:
            i=i+1
            pts_cloud[pointsIdx, -1] = (pts_cloud[pointsIdx,2] - localminZ)/(maxZ-localminZ)
            #pbar0.update(1)

    #pbar0.close()
    #print("\nLooping devoxelization: {}".format(i))
    return pts_cloud


def voxelize(pts_cloud:np.ndarray, voxels_size:list=None) -> dict:
    """Voxelize the points cloud into voxel cells.

    Args:
        pts_cloud (np.ndarray): [description]
        voxels_size (None): [description]

    Returs:
        dict: (x,y,z,ceilIndex) -> list
    """
    vSizeX, vSizeY, vSizeZ = voxels_size

    minX = pts_cloud[:,0].min()
    maxX = pts_cloud[:,0].max()
    minY = pts_cloud[:,1].min()
    maxY = pts_cloud[:,1].max()
    minZ = pts_cloud[:,2].min()
    maxZ = pts_cloud[:,2].max()

    numX = max(1, math.ceil((maxX-minX) / vSizeX))
    numY = max(1, math.ceil((maxY-minY) / vSizeY))
    numZ = 1

    numVoxel = numX * numY * numZ
    # print("Point Shape", pts_cloud.shape)
    # print("numX:{} numY:{} numZ:{}".format(numX, numY, numZ))
    # print(" xSize:{} ySize:{} zSize:{}".format(vSizeX, vSizeY, vSizeZ))
    voxels = {}
    #TODO: Parallelize voxels formation
    for idx, point in enumerate(pts_cloud):
        x = min(numX - 1, math.floor((point[0] - minX) / vSizeX))
        y = min(numY - 1, math.floor((point[1] - minY) / vSizeY))
        z = min(numZ - 1, math.floor((point[2] - minZ) / vSizeZ))

        celIdx = z * numX * numY + y * numX + x
        key = (x,y,z, 0) ## I don't remember why I do need the ceilIdx, but let's put a 0 to not break the code.
        if celIdx > numVoxel:
            print("ERROR: celIdx out of bounds {}, X:{}, Y:{}, Z:{}".format(celIdx, x, y, z))
        if key not in voxels:
            voxels[key] = []
        voxels[key].append(idx)

    return voxels


def global_trans_normalize(pc1,feat, model_type):
    pc=pc1.copy()
    intensity = np.array(feat[:, 0], dtype=np.float32)
    numret = np.array(feat[:, 1], dtype=np.float32)

    if (model_type =='B'):


        #create array to store AE; same dimension as original pts cloud after subsampling
        AE=np.zeros(intensity.shape, dtype=np.float32)
        #create array to store AE; same dimension as voxelized pts cloud
        #RE=np.zeros(features.shape, dtype=np.float32)

        #calculate absolute elevation
        maxZ = pc[:,2].max()
        AE = pc[:,2]/maxZ

        #run voxelization to get min local elevation within 1x1xinf grid
        voxels = voxelize(pc, voxels_size=(1,1,1))
        features = feature_extraction(voxels=voxels, pts_cloud=pc)
        #run devoxelization to map the minlocalZ back to each points in each voxel, and calculate RE at the same time
        recovered_pts_cloud=devoxelize(predictions=features, voxels=voxels, pts_cloud=pc, maxZ= maxZ)
        new_RE = recovered_pts_cloud[:,-1]


    #normalizing only intensity and number of return
    intensity_numret = np.hstack((intensity.reshape(-1,1), numret.reshape(-1,1)))

    #if t_normalize.get('method', None):
    #    method = t_normalize['method']
    #    if method == 'standard':
    #        normalized_feat = StandardScaler().fit_transform(intensity_numret)
    #    elif method == 'minmax':
    normalized_feat = MinMaxScaler().fit_transform(intensity_numret)

    if (model_type == 'B'):
        #combine intensity, number of return, AE and RE as new feature
        feat_new = np.hstack((normalized_feat, AE.reshape(-1,1), new_RE.reshape(-1,1)))
    else:
        feat_new = normalized_feat

    return feat_new