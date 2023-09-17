from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import tf_util as helper_tf_util
import time
import Lovasz_losses_tf as L


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
                self.saving_path = self.saving_path + '_' + dataset.name
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['b_xyz'] = flat_inputs[:num_layers]
            self.inputs['b_neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['b_sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['b_interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            
            self.inputs['m_xyz'] = flat_inputs[4 * num_layers:5 * num_layers]
            self.inputs['m_neigh_idx'] = flat_inputs[5 * num_layers:6 * num_layers]
            self.inputs['m_sub_idx'] = flat_inputs[6 * num_layers:7 * num_layers]
            self.inputs['m_interp_idx'] = flat_inputs[7 * num_layers:8 * num_layers]
            
            self.inputs['b_features'] = flat_inputs[8 * num_layers]
            self.inputs['m_features'] = flat_inputs[8 * num_layers + 1]
            self.inputs['labels'] = flat_inputs[8 * num_layers + 2]
            self.inputs['input_inds'] = flat_inputs[8 * num_layers + 3]
            self.inputs['cloud_inds'] = flat_inputs[8 * num_layers + 4]
            self.inputs['reorder_inds'] = flat_inputs[8 * num_layers + 5]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.loss_type = self.config.loss_function # wce, lovas, cbloss
            self.class_weights = DP.get_class_weights(dataset.num_per_class, self.loss_type, self.config.num_classes, self.config.use_custom_weights)
            makedirs(self.config.log_file_dir) if not exists(self.config.log_file_dir) else None
            self.Log_file = open(self.config.log_file_dir + '/' + config.log_file_name + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out
        b_feature = inputs['b_features']
        b_feature = tf.layers.dense(b_feature, 8, activation=None, name='fc0b')
        b_feature = tf.nn.leaky_relu(tf.layers.batch_normalization(b_feature, -1, 0.99, 1e-6, training=is_training))
        b_feature = tf.expand_dims(b_feature, axis=2)
        
        m_feature = inputs['m_features']
        m_feature = tf.layers.dense(m_feature, 8, activation=None, name='fc0m')
        m_feature = tf.nn.leaky_relu(tf.layers.batch_normalization(m_feature, -1, 0.99, 1e-6, training=is_training))
        m_feature = tf.expand_dims(m_feature, axis=2)

        # ###########################Encoder############################
        b_f_encoder_list = []
        m_f_encoder_list = []
        for i in range(self.config.num_layers):
            if i == 0:
                b_f_encoder_i, m_f_encoder_i = self.connection_block(b_feature, m_feature, self.config.n_point_connection[i], 4, 'Connection_Layer_' + str(i) , is_training)
            else:
                b_f_encoder_i, m_f_encoder_i = self.connection_block(b_feature, m_feature, self.config.n_point_connection[i], d_out[i-1], 'Connection_Layer_' + str(i) , is_training)
            
            b_f_encoder_i = self.dilated_res_block(b_f_encoder_i, inputs['b_xyz'][i], inputs['b_neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i) + '_b', is_training)
            m_f_encoder_i = self.dilated_res_block(m_f_encoder_i, inputs['m_xyz'][i], inputs['m_neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i)+ '_m', is_training)
            
            b_f_sampled_i = self.sample(b_f_encoder_i, inputs['b_sub_idx'][i])
            m_f_sampled_i = self.sample(m_f_encoder_i, inputs['m_sub_idx'][i])
            
            b_feature = b_f_sampled_i
            m_feature = m_f_sampled_i
            if i == 0:
                b_f_encoder_list.append(b_f_encoder_i)
            b_f_encoder_list.append(b_f_sampled_i)
            if i == 0:
                m_f_encoder_list.append(m_f_encoder_i)
            m_f_encoder_list.append(m_f_sampled_i)
        # ###########################Encoder############################

        b_feature = helper_tf_util.conv2d(b_f_encoder_list[-1], b_f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0_b',
                                        [1, 1], 'VALID', True, is_training)
        
        m_feature = helper_tf_util.conv2d(m_f_encoder_list[-1], m_f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0_m',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        b_f_decoder_list = []
        m_f_decoder_list = []
        for j in range(self.config.num_layers):
            b_f_interp_i = self.nearest_interpolation(b_feature, inputs['b_interp_idx'][-j - 1])
            m_f_interp_i = self.nearest_interpolation(m_feature, inputs['m_interp_idx'][-j - 1])
            
            
            b_f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([b_f_encoder_list[-j - 2], b_f_interp_i], axis=3),
                                                          b_f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j) + '_b', [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            m_f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([m_f_encoder_list[-j - 2], m_f_interp_i], axis=3),
                                                          m_f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j) + '_m', [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            b_feature = b_f_decoder_i
            m_feature = m_f_decoder_i
            b_f_decoder_list.append(b_f_decoder_i)
            m_f_decoder_list.append(m_f_decoder_i)
        # ###########################Decoder############################
        
        # ######################Feature Merging ########################
        #print(b_f_decoder_list[-1].get_shape)# [B, N, 1, F]
        base_overlapping_features = b_f_decoder_list[-1][:,:self.config.n_point_connection[0]:,:]
        medium_overlapping_features = m_f_decoder_list[-1][:,:self.config.n_point_connection[0],:,:]
        base_features = b_f_decoder_list[-1][:,self.config.n_point_connection[0]:,:]
        medium_features = m_f_decoder_list[-1][:,self.config.n_point_connection[0]:,:]
        features = self.mr_connection_block(base_overlapping_features, medium_overlapping_features, base_features, medium_features, self.config.num_points, 32, 'mr_connection',is_training)
        #features = tf.concat([overlapping_features, base_features, medium_features], 1)
        
        #features = helper_tf_util.conv2d(features, 64, [1, 1], 'feature_smoothing_mlp', [1, 1], 'VALID', True, is_training)
        
        f_layer_fc1 = helper_tf_util.conv2d(features, 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out

    def train(self, dataset):
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                #if m_iou > np.max(self.mIou_list):
                    # Save the best model
                snapshot_directory = join(self.saving_path, 'snapshots')
                makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)
                
                if self.config.save_preset_epoch:
                    if self.training_epoch == self.config.preset_1:
                        snapshot_directory = join(self.saving_path, 'preset_1')
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)

                    elif self.training_epoch == self.config.preset_2:
                        snapshot_directory = join(self.saving_path, 'preset_2')
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                    
                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0
        
        base_gt_classes = [0 for _ in range(self.config.num_classes)]
        base_positive_classes = [0 for _ in range(self.config.num_classes)]
        base_true_positive_classes = [0 for _ in range(self.config.num_classes)]
        base_val_total_correct = 0
        base_val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if self.config.show_base_only:
                    base_pred = np.reshape(pred, (self.config.val_batch_size, -1))[:,:self.config.num_points*4//7].flatten()
                    base_labels = np.reshape(labels, (self.config.val_batch_size, -1))[:,:self.config.num_points*4//7].flatten()
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                    if self.config.show_base_only:
                        base_pred_valid = base_pred
                        base_labels_valid = base_labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)
                    
                    if self.config.show_base_only:
                        invalid_idx = np.where(base_labels == self.config.ignored_label_inds)[0]
                        base_labels_valid = np.delete(base_labels, invalid_idx)
                        base_labels_valid = base_labels_valid - 1
                        base_pred_valid = np.delete(base_pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)
                
                if self.config.show_base_only:
                    base_correct = np.sum(base_pred_valid == base_labels_valid)
                    base_val_total_correct += base_correct
                    base_val_total_seen += len(base_labels_valid)

                    base_conf_matrix = confusion_matrix(base_labels_valid, base_pred_valid, np.arange(0, self.config.num_classes, 1))
                    base_gt_classes += np.sum(base_conf_matrix, axis=1)
                    base_positive_classes += np.sum(base_conf_matrix, axis=0)
                    base_true_positive_classes += np.diagonal(base_conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)
            
        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)
        
        if self.config.show_base_only:
            base_iou_list = []
            for n in range(0, self.config.num_classes, 1):
                base_iou = base_true_positive_classes[n] / float(base_gt_classes[n] + base_positive_classes[n] - base_true_positive_classes[n])
                base_iou_list.append(base_iou)
            base_mean_iou = sum(base_iou_list) / float(self.config.num_classes)
            
            log_out('base eval accuracy: {}'.format(base_val_total_correct / float(base_val_total_seen)), self.Log_file)
            log_out('base mean IOU:{}'.format(base_mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        s = '{:5.2f} | '.format(mean_iou)
        if self.config.show_base_only:
            base_mean_iou = 100 * base_mean_iou
            base_s = '{:5.2f} | '.format(base_mean_iou)
            
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        if self.config.show_base_only:
            for base_IoU in base_iou_list:
                base_s += '{:5.2f} '.format(100 * base_IoU)
            
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        if self.config.show_base_only:
            log_out(base_s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        
        
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        if self.loss_type == 'lovas':
            logits = tf.reshape(logits, [-1, self.config.num_classes])  # [-1, n_class]
            probs = tf.nn.softmax(logits, axis=-1)  # [-1, class]
            labels = tf.reshape(labels, [-1])
            lovas_loss = L.lovasz_softmax(probs, labels, 'present')
            output_loss = output_loss + lovas_loss
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    def connection_block(self, points1, points2, n_point, d_out, name, is_training):
        print('before connection', points1.get_shape)
        connected_2, separated_2 = self.base_medium_splitter(points2, n_point)
        connected_1, separated_1 = points1[:,:n_point], points1[:,n_point:]
        feature = tf.concat([connected_1, connected_2], -1)#B,N,1,2F
        feature = helper_tf_util.conv2d(feature, d_out*2, [1, 1], name + 'mlp1', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        connected_1, connected_2 = self.gfsae_CN(feature, d_out*2, is_training, name)
        #print(b_f_decoder_list[-1].get_shape)
        points1 = tf.concat([connected_1, separated_1], 1)
        points2 = tf.concat([connected_2, separated_2], 1)
        print('after connection',points1.get_shape)
        return points1, points2
    
    def mr_connection_block(self, points1, points2, base_features, medium_features, n_point, d_out, name, is_training):
        feature = tf.concat([points1, points2], -1)
        feature = helper_tf_util.conv2d(feature, d_out, [1, 1], name + 'mlp1', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        base_features = helper_tf_util.conv2d(base_features, d_out, [1, 1], name + 'mlpb', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        medium_features = helper_tf_util.conv2d(medium_features, d_out, [1, 1], name + 'mlpm', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        
        feature = tf.concat([feature, base_features, medium_features], 1)
        feature = helper_tf_util.conv2d(feature, d_out, [1, 1], name + 'mlpc', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        
        z = tf.reduce_mean(feature, axis = 1)
        z = tf.nn.leaky_relu(tf.layers.dense(z, d_out, activation=None, name= name + 'fc0'))
        z = tf.nn.sigmoid(tf.layers.dense(z, d_out, activation=None, name= name + 'fc1'))#(B,f,1)
        
        feature = tf.squeeze(feature, axis=2)
        feature = tf.multiply(tf.tile(z, (1,n_point, 1)), feature)
        feature = tf.expand_dims(feature, axis=2)
        
        return feature

    def gfsae_CN(self, feature, d, is_training, name) :
        """
        graph self attention
        """
        #d = feature.get_shape()[-1].value
        # Self-attenctive encoding layer
        gfsae_fc1 = helper_tf_util.conv2d(feature, d//2, [1, 1], name+'_gfsae_fc1', [1, 1], 'VALID', True, is_training)  # B,N,1,d/2
        gfsae_fc2 = helper_tf_util.conv2d(gfsae_fc1, d, [1, 1], name+'_gfsae_fc2', [1, 1], 'VALID', True, is_training)  # B,N,1,d
        gfsae_fc3_b = helper_tf_util.conv2d(gfsae_fc2, 2*d, [1, 1], name+'_gfsae_fc3_b', [1, 1], 'VALID', True, is_training)  # B,N,1,2d
        gfsae_fc3_m = helper_tf_util.conv2d(gfsae_fc2, 2*d, [1, 1], name+'_gfsae_fc3_m', [1, 1], 'VALID', True, is_training)  # B,N,1,2d
        gfsae_fc3_sfmx_b = tf.nn.softmax(gfsae_fc3_b, axis=1) # B,N,1,2d
        gfsae_fc3_sfmx_m = tf.nn.softmax(gfsae_fc3_m, axis=1) # B,N,1,2d
        
        gfsae_f_att_b = gfsae_fc3_b * gfsae_fc3_sfmx_b # B,N,1,2d
        gfsae_f_att_m = gfsae_fc3_m * gfsae_fc3_sfmx_m # B,N,1,2d
        gfsae_f_se_star_b = tf.reduce_sum(gfsae_f_att_b, axis=1) # B,1,2d
        gfsae_f_se_star_m = tf.reduce_sum(gfsae_f_att_m, axis=1) # B,1,2d
        
        gfsae_f_se_b = tf.tile(tf.expand_dims(gfsae_f_se_star_b, axis=1), [1, tf.shape(gfsae_f_att_b)[1], 1, 1]) # B,N,1,2d
        gfsae_f_se_m = tf.tile(tf.expand_dims(gfsae_f_se_star_m, axis=1), [1, tf.shape(gfsae_f_att_m)[1], 1, 1]) # B,N,1,2d
        gfsae_b = tf.concat([gfsae_fc2, gfsae_fc3_b, gfsae_f_se_b], axis = 3) # B,N,1,5d
        gfsae_m = tf.concat([gfsae_fc2, gfsae_fc3_m, gfsae_f_se_m], axis = 3) # B,N,1,5d
        
        
        # Channel enchancement layer
        gfsae_f_ce_b = helper_tf_util.conv2d(gfsae_b, 2*d, [1, 1], name+'_gfsae_b', [1, 1], 'VALID', True, is_training) # B,N,1,2d
        gfsae_f_ce_m = helper_tf_util.conv2d(gfsae_m, 2*d, [1, 1], name+'_gfsae_m', [1, 1], 'VALID', True, is_training) # B,N,1,2d
        gfsae_f_ce_sfmx_b = tf.nn.softmax(gfsae_f_ce_b, axis=1) # B,N,1,2d
        gfsae_f_ce_sfmx_m = tf.nn.softmax(gfsae_f_ce_m, axis=1) # B,N,1,2d
        gfsae_f_att2_b = gfsae_f_ce_b * gfsae_f_ce_sfmx_b # B,N,1,2d
        gfsae_f_att2_m = gfsae_f_ce_m * gfsae_f_ce_sfmx_m # B,N,1,2d
        gfsae_f_se2_star_b = tf.reduce_sum(gfsae_f_att2_b, axis=1) # B,1,2d
        gfsae_f_se2_star_m = tf.reduce_sum(gfsae_f_att2_m, axis=1) # B,1,2d
        gfsae_v_raw_b = helper_tf_util.conv2d(tf.expand_dims(gfsae_f_se2_star_b, axis=1), d, [1, 1],
                                           name + '_gfsae_v_raw_b', [1, 1], 'VALID', True, is_training, activation_fn=None)  # B,1,1,d
        gfsae_v_raw_m = helper_tf_util.conv2d(tf.expand_dims(gfsae_f_se2_star_m, axis=1), d, [1, 1],
                                           name + '_gfsae_v_raw_m', [1, 1], 'VALID', True, is_training, activation_fn=None)  # B,1,1,d
        gfsae_v_att_b = tf.sigmoid(gfsae_v_raw_b) # B,1,1,d
        gfsae_v_att_m = tf.sigmoid(gfsae_v_raw_m) # B,1,1,d
        feature_b = feature + feature * gfsae_v_att_b # B,N,1,d
        feature_m = feature + feature * gfsae_v_att_m # B,N,1,d
        print(feature_b.get_shape())
        
        return feature_b, feature_m
    
    @staticmethod
    def base_medium_splitter(points, base_n):
        base_points = points[:, :base_n,:]
    
        medium_points = points[:,base_n:,:]
        return base_points, medium_points
    
    @staticmethod
    def sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features
    

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
