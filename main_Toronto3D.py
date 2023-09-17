from os.path import join, exists
from EyeNet import Network
from tester_Toronto3D import ModelTester
from helper_ply import read_ply
from tool import DataProcessing as DP
from tool import ConfigToronto3D as cfg
import tensorflow as tf
import numpy as np
import pickle, argparse, os


class Toronto3D:
    def __init__(self, mode='train'):
        self.name = 'Toronto3D'
        self.path = cfg.data_set_dir
        self.label_to_names = {0: 'unclassified',
                               1: 'Ground',
                               2: 'Road marking',
                               3: 'Natural',
                               4: 'Building',
                               5: 'Utility line',
                               6: 'Pole',
                               7: 'Car',
                               8: 'Fence'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])

        self.full_pc_folder = join(self.path, 'original_files')

        # Initial training-validation-testing files
        self.train_files = ['L001', 'L003', 'L004']
        self.val_files = ['L002']
        self.test_files = ['L002']

        self.val_split = 3

        self.train_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.train_files]
        self.val_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.val_files]
        self.test_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.test_files]
        
        print(self.train_files)

        # Initiate containers
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}

        self.load_sub_sampled_clouds(cfg.sub_grid_size, mode)
        
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def load_sub_sampled_clouds(self, sub_grid_size, mode):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        if mode == 'test':
            files = self.test_files
        else: 
            files = np.hstack((self.train_files, self.val_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if mode == 'test':
                cloud_split = 'test'
            else:
                if file_path in self.val_files:
                    cloud_split = 'validation'
                else:
                    cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)
            # read RGB / intensity accoring to configuration
            if cfg.use_rgb and cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'], data['intensity'])).T
            elif cfg.use_rgb and not cfg.use_intensity:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            elif not cfg.use_rgb and cfg.use_intensity:
                sub_colors = data['intensity'].reshape(-1,1)
            else:
                sub_colors = np.ones((data.shape[0],1))
            #if cloud_split == 'test':
            #    sub_labels = None
            #else:
            sub_labels = data['class']

            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)
                
            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            #if cloud_split in ['training', 'validation']:
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            # Get test re_projection indices
            if cloud_split == 'test':
                print('\nPreparing reprojection indices for {}'.format(cloud_name))
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]

        print('finished')
        return

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        
        # assign number of features according to input
        n_features = 1 # use xyz only by default
        if cfg.use_rgb and cfg.use_intensity:
            n_features = 4
        elif cfg.use_rgb and not cfg.use_intensity:
            n_features = 3

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

        # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                #query_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
                #collect points for base receptive field
                if len(points) < cfg.num_points * 4 //7:
                    base_queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    base_queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points * 4 //7)[1][0]
                

                # Shuffle index
                #query_idx = DP.shuffle_idx(query_idx)
                base_queried_idx = DP.shuffle_idx(base_queried_idx)

                # Get corresponding points and colors based on the index
                #queried_pc_xyz = points[query_idx]
                #queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                #queried_pc_colors = self.input_colors[split][cloud_idx][query_idx]
                base_queried_pc_xyz = points[base_queried_idx]
                base_queried_pc_xyz[:, 0:2] = base_queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                base_queried_pc_colors = self.input_colors[split][cloud_idx][base_queried_idx]  
                
                #if split == 'test':
                    #queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    #queried_pt_weight = 1
                #    base_queried_pc_labels = np.zeros(base_queried_pc_xyz.shape[0])
                #    base_queried_pt_weight = 1
                #else:
                    #queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    #queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    #queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])
                    #base_queried_pc_colors = self.input_colors[split][cloud_idx][base_queried_idx]  
                base_queried_pc_labels = self.input_labels[split][cloud_idx][base_queried_idx]
                base_queried_pc_labels = np.array([self.label_to_idx[l] for l in base_queried_pc_labels])

                # Collect points and colors for medium receptive field
                base_dists = np.sum(np.square((points[base_queried_idx] - pick_point).astype(np.float32)), axis=1)
                base_r = np.sqrt(np.max(base_dists))
                
                medium_r = base_r*2
                ind = self.input_trees[split][cloud_idx].query_radius(pick_point, r=medium_r)[0]
                medium_queried_idx = np.setdiff1d(ind, base_queried_idx,assume_unique=True)
                
                medium_queried_idx = DP.shuffle_idx(medium_queried_idx)[:cfg.num_points * 3 //7]
                
                medium_queried_pc_xyz = points[medium_queried_idx]
                medium_queried_pc_xyz[:, 0:2] = medium_queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                medium_queried_pc_colors = self.input_colors[split][cloud_idx][medium_queried_idx]
                
                #if split == 'test':
                    #queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
                    #queried_pt_weight = 1
                #    medium_queried_pc_labels = np.zeros(medium_queried_pc_xyz.shape[0])
                #    medium_queried_pt_weight = 1
                #else:
                    #queried_pc_labels = self.input_labels[split][cloud_idx][query_idx]
                    #queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
                    #queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])
                medium_queried_pc_labels = self.input_labels[split][cloud_idx][medium_queried_idx]
                medium_queried_pc_labels = np.array([self.label_to_idx[l] for l in medium_queried_pc_labels])
                    
                    
                
                if len(points) < cfg.num_points * 4 //7:
                    base_queried_pc_xyz, base_queried_pc_colors, base_queried_idx, base_queried_pc_labels = \
                        DP.data_aug(base_queried_pc_xyz, 
                                    base_queried_pc_colors, 
                                    base_queried_pc_labels,
                                    base_queried_idx, 
                                    cfg.num_points * 4 //7)
                    
                if len(medium_queried_pc_xyz) < cfg.num_points * 3 //7:
                    medium_queried_pc_xyz, medium_queried_pc_colors, medium_queried_idx, medium_queried_pc_labels = \
                        DP.data_aug(medium_queried_pc_xyz,
                                    medium_queried_pc_colors,
                                    medium_queried_pc_labels,
                                    medium_queried_idx,
                                    cfg.num_points * 3 //7)
                    
                #concatenate base and medium indexes
                query_idx = np.concatenate((base_queried_idx, medium_queried_idx))

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                #combine medium and base points
                queried_pc_xyz = np.concatenate((base_queried_pc_xyz, medium_queried_pc_xyz), axis = 0)
                queried_pc_colors = np.concatenate((base_queried_pc_colors, medium_queried_pc_colors), axis = 0)
                queried_pc_labels = np.concatenate((base_queried_pc_labels, medium_queried_pc_labels), axis = 0)
                
                if True:
                    yield (queried_pc_xyz,
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, n_features], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self):
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            if not cfg.use_rgb and not cfg.use_intensity:
                batch_features = batch_xyz
            else :
                batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            #separate points to base and medium receptive field
            b_batch_xyz, m_batch_xyz = batch_xyz[:,:cfg.num_points * 4 //7,:], batch_xyz[:,cfg.num_points * 4 //7:,:]
            b_batch_features, m_batch_features = batch_features[:,:cfg.num_points * 4 //7,:], batch_features[:,cfg.num_points * 4 //7:,:]
            b_batch_xyz_opp = b_batch_xyz
            m_batch_xyz_opp = m_batch_xyz
            b_input_points = []
            b_input_neighbors = []
            b_input_pools = []
            b_input_up_samples = []


            m_input_points =[]
            m_input_neighbors = []
            m_input_pools = []
            m_input_up_samples = []
            
            #currently it always assume the last subsampling ratio to be 4. Is it even possible to use 2 like original RandLA-Net Implementation?
            for i in range(cfg.num_layers):
                #processing base points
                neigh_idx = tf.py_func(DP.knn_search, [b_batch_xyz, b_batch_xyz, cfg.k_n[i]], tf.int32)
                sub_points = b_batch_xyz[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                if i == 0:
                    sub_features = b_batch_features[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                    m_batch_features = tf.concat((sub_features,m_batch_features), 1)
                pool_i = neigh_idx[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, b_batch_xyz, 1], tf.int32)
                
                b_input_points.append(b_batch_xyz)
                b_input_neighbors.append(neigh_idx)
                b_input_pools.append(pool_i)
                b_input_up_samples.append(up_i)
                
                
                if cfg.sub_sampling_ratio[i] == cfg.connection_ratio:
                    b_batch_xyz = sub_points
                else:
                    addtional_sampling_ratio = cfg.connection_ratio//cfg.sub_sampling_ratio[i]
                    b_batch_xyz = sub_points[:, :tf.shape(sub_points)[1] // addtional_sampling_ratio, :]
                
                #processing medium points
                m_input_data = tf.concat((b_batch_xyz, m_batch_xyz), 1)
                m_neigh_idx = tf.py_func(DP.knn_search, [m_input_data, m_input_data, cfg.k_n[i]], tf.int32)
                
                m_b_neigh_idx = m_neigh_idx[:,:tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_m_neigh_idx = m_neigh_idx[:,tf.shape(b_batch_xyz)[1]: tf.shape(b_batch_xyz)[1] + tf.shape(m_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_pool_i = tf.concat((m_b_neigh_idx, m_m_neigh_idx), 1)
                
                b_sub_points = b_batch_xyz[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_batch_xyz = m_batch_xyz[:, :tf.shape(m_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_sub_points = tf.concat((b_sub_points, m_batch_xyz), 1)
                
                m_up_i = tf.py_func(DP.knn_search, [m_sub_points, m_input_data, 1], tf.int32)
                
                m_input_points.append(m_input_data)#[12288,3072, 768, 192, 48]
                m_input_neighbors.append(m_neigh_idx)
                m_input_pools.append(m_pool_i)
                m_input_up_samples.append(m_up_i)
                ##########################################################################################
            
            opp = b_batch_xyz_opp[:, tf.shape(b_batch_xyz_opp)[1] // cfg.sub_sampling_ratio[0]:, :]
            cat_xyz = tf.concat((b_input_points[1],opp, m_batch_xyz_opp), axis = 1)
            reorder_idx = tf.py_func(DP.knn_search, [cat_xyz, batch_xyz, 1], tf.int32)
            
            
            input_list = b_input_points + b_input_neighbors + b_input_pools + b_input_up_samples + m_input_points + m_input_neighbors + m_input_pools + m_input_up_samples
            input_list += [b_batch_features, m_batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, reorder_idx]

            return input_list

        return tf_map

    # data augmentation
    @staticmethod
    def tf_augment_input(inputs):
        xyz = inputs[0]
        features = inputs[1]
        theta = tf.random_uniform((1,), minval=0, maxval=2 * np.pi)
        # Rotation matrices
        c, s = tf.cos(theta), tf.sin(theta)
        cs0 = tf.zeros_like(c)
        cs1 = tf.ones_like(c)
        R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
        stacked_rots = tf.reshape(R, (3, 3))

        # Apply rotations
        transformed_xyz = tf.reshape(tf.matmul(xyz, stacked_rots), [-1, 3])
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = tf.random_uniform((1, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((1, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([1, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.tile(s, [tf.shape(transformed_xyz)[0], 1])

        # Apply scales
        transformed_xyz = transformed_xyz * stacked_scales

        noise = tf.random_normal(tf.shape(transformed_xyz), stddev=cfg.augment_noise)
        transformed_xyz = transformed_xyz + noise
        stacked_features = tf.concat([transformed_xyz, features], axis=-1)
        return stacked_features

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        gen_function_test, _, _ = self.get_batch_gen('test')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)
        self.test_init_op = iter.make_initializer(self.batch_test_data)


    def init_test_pipeline(self):
        print('Initiating testing pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function_test,gen_types, gen_shapes = self.get_batch_gen('test')

        self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
        self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping()
        self.batch_test_data = self.batch_test_data.map(map_func=map_func)
        self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_test_data.output_types, self.batch_test_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.test_init_op = iter.make_initializer(self.batch_test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    parser.add_argument('--test_eval', type=bool, default=False, help='evaluate test result on L002')
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    Mode = FLAGS.mode
    dataset = Toronto3D(mode=Mode)
    
    if Mode == 'train':
        dataset.init_input_pipeline()
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        dataset.init_test_pipeline()
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snap = FLAGS.model_path
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)

    else:
        raise ValueError('mode not supported')