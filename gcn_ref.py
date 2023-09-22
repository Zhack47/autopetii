import SimpleITK
import numpy as np
import nibabel as nib
from scipy import ndimage
from os.path import join
import scipy.sparse as sp
import torch
import os


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def connect_n6_krandom(ref, voxel_node, node_voxel, working_nodes, k_random, weighting, args):
    edges = []
    labels = []
    num_nodes = node_voxel.shape[0]
    tabu_list = {}  # A list to avoid duplicated elements in the adjacency matrix.
    nodes_complete = {}  # A list counting how many neighbors a node already has.
    valid_nodes = np.array(np.where(working_nodes > 0))
    valid_nodes = np.transpose(valid_nodes)

    for node_idx in range(num_nodes):
        y, x, z = node_voxel[node_idx]  # getting the 3d position for current node
        labels.append(ref[y, x, z])  # Labels come from the CNN prediction
        #   Basic n6 connectivity
        for axis in range(3):
            axisy = int(axis == 0)
            axisx = int(axis == 1)
            axisz = int(axis == 2)
            for ne in [-1, 1]:
                neighbor = y + axisy*ne, x + axisx*ne, z + axisz*ne
                if neighbor not in voxel_node:
                    continue
                ne_idx = voxel_node[neighbor]
                if (node_idx, ne_idx) not in tabu_list and (ne_idx, node_idx) not in \
                        tabu_list:
                    tabu_list[(node_idx, ne_idx)] = 1  # adding the edge to the tabu list
                    weighting.weights_for((y, x, z), neighbor, args)  # computing the weight for the current pair.
                    weighting.weights_for(neighbor, (y, x, z), args)  # computing the weight for the current pair.
                    edges.append([node_idx, ne_idx])
#                   Adding the edge in the opposite direction.
                    edges.append([ne_idx, node_idx])
#       Generating random connections to current node.
        for j in range(k_random):
            valid_neigh = False
            if node_idx not in nodes_complete:
                nodes_complete[node_idx] = 0
            elif nodes_complete[node_idx] == k_random:
                break

            while not valid_neigh:
                lu_idx = np.random.randint(low=0, high=num_nodes)  # we look for a random node.
                yl, xl, zl = valid_nodes[lu_idx]  # getting the euclidean coordinates for the voxel.
                lu_idx = voxel_node[yl, xl, zl]  # getting the node index.
                if lu_idx not in nodes_complete:
                    nodes_complete[lu_idx] = 0
                    valid_neigh = True
                elif nodes_complete[lu_idx] < k_random:
                    valid_neigh = True

            if not (node_idx, lu_idx) in tabu_list and not (lu_idx, node_idx) in tabu_list \
                    and node_idx != lu_idx:  # checking if the edge was already generated
                weighting.weights_for((y, x, z), (yl, xl, zl), args)  # computing the weight for the current pair.
                weighting.weights_for((yl, xl, zl), (y, x, z), args)
                tabu_list[(node_idx, lu_idx)] = 1
                edges.append([node_idx, lu_idx])
                #  Adding the weight in the opposite direction
                edges.append([lu_idx, node_idx])
                #  Increasing the amount of neighbors connected to each node
                nodes_complete[node_idx] += 1
                nodes_complete[lu_idx] += 1
    edges = np.asarray(edges, dtype=int)
    pp_args = {
        "edges": edges,
        "num_nodes": num_nodes
    }
    weighting.post_process(pp_args)  # Applying weight post-processing, e.g. normalization
    weights = weighting.get_weights()
    edges, weights, _ = sparse_to_tuple(weights)

    return edges, weights, np.asarray(labels, dtype=np.float32), num_nodes


def get_connect_func(cf_id):
    if cf_id == 1:
        return connect_n6_krandom

    return None



def add_feature(add_vol, ft_vol):
    axis = len(add_vol.shape)
    new_fts = np.expand_dims(add_vol, axis=axis)
    ret_fts = np.concatenate((ft_vol, new_fts), axis=axis)
    return ret_fts

def map_voxel_nodes(shape, include_nodes):
    ys, xs, zs = shape
    N = np.sum(include_nodes.astype(int))
    voxel_node = {}
    node_voxel = np.zeros(shape=(N, 3), dtype=int)
    node_index = 0
    for z in range(zs):
        for y in range(ys):
            for x in range(xs):
                if not include_nodes[y, x, z]:
                    continue
                node_voxel[node_index] = [y, x, z]
                voxel_node[y, x, z] = node_index
                node_index += 1
    return voxel_node, node_voxel


def graph_fts(fts, node_voxel):
    N = node_voxel.shape[0]
    K = fts.shape[3]  # number of features per node.
    ft_mat = np.zeros(shape=(N, K), dtype=np.float32)
    for node_idx in range(N):
        y, x, z = node_voxel[node_idx]
        ft_mat[node_idx, :] = fts[y, x, z, :]
    return ft_mat


def generate_mask(unc_vol, node_voxel, th=0):
    num_nodes = node_voxel.shape[0]
    mask = np.zeros(shape=(num_nodes, 1), dtype=np.float32)
    for node_idx in range(num_nodes):
        y, x, z = node_voxel[node_idx]
        mask[node_idx] = float(unc_vol[y, x, z] > th)
    return mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def generate_components2(patient_path, probability, entropy, seg_vol, bin_entropy, pet_volume, dilated_vol, connect_funct, weight):
    num_vox = float(pet_volume.shape[0] * pet_volume.shape[1] * pet_volume.shape[2])
    vmu = pet_volume.astype(np.float32).sum() / num_vox
    vvar = np.sum((pet_volume.astype(np.float32) - vmu) ** 2) / num_vox

    fts = np.array(pet_volume, dtype=np.float32)
    fts = (fts - vmu) / vvar
    fts = np.expand_dims(fts, 3)

    fts = add_feature(probability, fts)
    fts = add_feature(entropy, fts)

    #valid_nodes = np.load(valid_path)
    voxel_node, node_voxel = map_voxel_nodes(pet_volume.shape, dilated_vol.astype(bool))
    ft_graph = graph_fts(fts, node_voxel)  # convert the feature vol to a graph representation

    args = {
        "volume": (pet_volume.astype(np.float32) - vmu) / vvar,
        "prediction": seg_vol,
        "probability": probability,
        "uncertainty": bin_entropy,
        "entropy_map": entropy,
        "features": fts
    }

    graph, weights, lb, N = get_connect_func(connect_funct)(ref=seg_vol, voxel_node=voxel_node,
                                                                node_voxel=node_voxel, working_nodes=dilated_vol,
                                                                k_random=16, weighting=get_weighting_func(weight),
                                                                args=args)
    mask = generate_mask(bin_entropy, node_voxel)  # Uncertainty mask
    # Volume ground truth are represented as nodes in the graph (reference graph)

    np.save(join(patient_path, "graph.npy"), graph)
    np.save(join(patient_path, "weights.npy"), weights)
    np.save(join(patient_path, "features.npy"), ft_graph)
    np.save(join(patient_path, "labels.npy"), lb)  # Node labels from CNN prediction
    np.save(join(patient_path, "mask.npy"), mask)  # Elements that should not be part of training (uncertain points)
    np.save(join(patient_path, "dilated.npy"), dilated_vol)  # Elements that should not be part of training (uncertain points)
    np.save(join(patient_path, "bin_entropy.npy"), bin_entropy)  # Elements that should not be part of training (uncertain points)

    info = {
        "N" : N,
        "total_edges": N,
        "graph_shape": graph.shape,
        "weight_shape": weights.shape,
        "ft_shape": ft_graph.shape,
        "train_labels_shape": lb.shape,
        "mask_uncertainty_shape": mask.shape,
        "num_nodes" : np.sum(dilated_vol),
        "num_uncertainty_nodes": np.sum(mask),
        "num_certainty_nodes": N - np.sum(mask),
        "num_positive_samples": np.sum(lb[np.where(mask == 0)[0]] == 1),
        "num_negative_samples": np.sum(lb[np.where(mask == 0)[0]] == 0)
    }
    return info


def bounding_cube(vol, offset=0):
    a = np.where(vol != 0)
    bbox = np.min(a[0]), np.min(a[1]), np.min(a[2]) - offset, \
        np.max(a[0]) + 1, np.max(a[1]) + 1, np.max(a[2]) + 1 + offset
    return bbox


class BasicWeighting:
    def __init__(self, w_id):
        self.description = "All edges are weighted as 1"
        self.id = w_id
        self.weights = []

    def weights_for(self, idx1, idx2, args):
        self.weights.append(1)

    def post_process(self, args=None):
        self.weights = np.asarray(self.weights, dtype=np.float32)
        num_nodes = args["num_nodes"]
        w1 = sp.coo_matrix((self.weights, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        self.weights = w1

    def get_weights(self):
        return self.weights

    def get_id(self):
        return self.id

    @property
    def get_description(self):
        return self.description


class Weighting1(BasicWeighting):
    def __init__(self, w_id):
        super(Weighting1, self).__init__(w_id=w_id)
        self.description = "l*div + e(int) + e(pos)"
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []

    def weights_for(self, idx1, idx2, args):
        prob1 = args["probability"][idx1]
        prob2 = args["probability"][idx2]
        int1 = args["volume"][idx1]
        int2 = args["volume"][idx2]
        ny, nx, nz = args["volume"].shape
        dim_array = np.array([ny, nx, nz], dtype=np.float32)
        pos1 = np.array(idx1,dtype=np.float32) / dim_array
        pos2 = np.array(idx2, dtype=np.float32) / dim_array
#       Computing the weight
        int_diff = int1 - int2
        pos_diff = pos1 - pos2
        intensity = np.sum(int_diff * int_diff)
        space = np.sum(pos_diff * pos_diff)
        p = prob1 - prob2
        delta = 1.0e-15
        lambd = p * (np.log2(prob1 / (prob2 + delta) + delta) - np.log2((1 - prob1) / ((1 - prob2) + delta) + delta))
        self.weights1.append(lambd)
        self.weights2.append(intensity)
        self.weights3.append(space)

    def post_process(self, args=None):
        self.weights1 = np.asarray(self.weights1, dtype=np.float32)
        self.weights2 = np.asarray(self.weights2, dtype=np.float32)
        self.weights3 = np.asarray(self.weights3, dtype=np.float32)
        num_nodes = args["num_nodes"]
        ne = float(self.weights1.shape[0])
        muw2 = self.weights2.sum() / ne
        muw3 = self.weights3.sum() / ne

        sig2 = 2 * np.sum((self.weights2 - muw2) ** 2) / ne
        sig3 = 2 * np.sum((self.weights3 - muw3) ** 2) / ne

        self.weights2 = np.exp(-self.weights2 / sig2)
        self.weights3 = np.exp(-self.weights3 / sig3)

        w1 = sp.coo_matrix((self.weights1, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w2 = sp.coo_matrix((self.weights2, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))
        w3 = sp.coo_matrix((self.weights3, (args["edges"][:, 0], args["edges"][:, 1])), shape=(num_nodes, num_nodes))

        self.weights = 0.5 * w1 + w2 + w3


def get_weighting_func(w_id):
    if w_id == 0:
        return BasicWeighting(w_id)
    if w_id == 1:
        return Weighting1(w_id=1)
    return None



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def load_data_infer(patient_path):
    val_portion = 0.2

    graph_path = join(patient_path, "graph.npy")
    weights_path = join(patient_path, "weights.npy")
    features_path = join(patient_path, "features.npy")
    labels_path = join(patient_path, "labels.npy")
    mask_path = join(patient_path, "mask.npy")  # Elements that should not be part of training but used for testing

    try:
        graph = np.load(graph_path)
        weights = np.load(weights_path)
        features = np.load(features_path)
        test_mask = np.load(mask_path)  # Elements that should not be part of training but used for testing
    except FileNotFoundError:
        return None

    full_mask = 1 - test_mask   # Elements that will be used for training the model

    labels = np.load(labels_path)  # All the predicted (from model) labels are included here.
    num_nodes = labels.shape[0]
    adj = sp.coo_matrix((weights, (graph[:, 0], graph[:, 1])), shape=(num_nodes, num_nodes))
    features = sp.coo_matrix(features)
    working_nodes = np.where(full_mask != 0)[0]
    random_arr = np.random.uniform(low=0, high=1, size=working_nodes.shape)

    features = normalize(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))



    idx_train = working_nodes[random_arr > val_portion]
    idx_val = working_nodes[random_arr <= val_portion]
    idx_test = np.where(test_mask != 0)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test[0])

    return adj, features, labels, idx_train, idx_val, idx_test



def reconstruct_from_n6(ft_mat, map_vector, shape, dtype=np.uint8):
    ys, xs, zs = shape
    N = map_vector.shape[0]
    rec_vol = np.zeros(shape=(ys, xs, zs), dtype=dtype)
    for i in range(N):
        y, x, z = map_vector[i]
        rec_vol[y, x, z] = dtype(ft_mat[i])
    return rec_vol



def gcn_inference2(model, sample_seg, sample_probs, mha_input_path):
    save_dir_graph = "../hot/autopet_graphs"
    img = SimpleITK.ReadImage(sample_seg)
    SimpleITK.WriteImage(img, "seg.nii.gz", True)
    nii_seg = nib.load("seg.nii.gz")
    np_seg = np.array(nii_seg.dataobj)
    img = SimpleITK.ReadImage(mha_input_path)
    SimpleITK.WriteImage(img, "tep.nii.gz", True)
    nii_pet_vol = nib.load("tep.nii.gz")
    np_pet_vol = np.array(nii_pet_vol.dataobj)
    if np.sum(np_seg):
        roi_limits = bounding_cube(np_seg)
        probabilities = np.transpose(np.load(sample_probs)["probabilities"][1],
                                     axes=(2, 1, 0))
        entropy = -probabilities * np.log2(probabilities + 1.0e-15) - (1.0 - probabilities) * np.log2(
            1.0 - probabilities + 1.0e-15)
        entropy_th = .8
        roi_prediction = np_seg[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
        roi_prob = probabilities[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
        roi_entropy = entropy[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
        roi_seg = np_seg[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]
        roi_pet = np_pet_vol[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]]

        bin_entropy = (roi_entropy > entropy_th).astype(np.uint8)
        kernel = np.ones(shape=(5, 7, 7), dtype=bool)
        dilated = ndimage.binary_dilation(bin_entropy, structure=kernel).astype(np.uint8)
        dilated = ((dilated + roi_seg) > 0).astype(int)

        expanded_bin_entropy = np.zeros(shape=np_seg.shape)
        expanded_bin_entropy[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4],
        roi_limits[2]:roi_limits[5]] = bin_entropy

        cf = 1
        w = 1
        graph_info = generate_components2(join(save_dir_graph, sample_seg.replace(".nii.gz", "").split(os.sep)[-1]), roi_prob, roi_entropy,
                                         roi_prediction, bin_entropy, roi_pet,
                                         dilated, cf, w)
        model.eval()
        adj, feats, _, _, _, _ = load_data_infer(patient_path=join(save_dir_graph, sample_seg.replace(".nii.gz", "").split(os.sep)[-1]))
        print(feats.shape)
        print(adj.shape)
        output = model.to("cuda")(feats.to("cuda"), adj.to("cuda"))

        graph_predictions = (output > .5).cpu().numpy().astype(float)

        valid_nodes = dilated
        roi_vol_shape = (roi_limits[3]-roi_limits[0], roi_limits[4]-roi_limits[1], roi_limits[5]-roi_limits[2])

        voxel_node, node_voxel = map_voxel_nodes(roi_vol_shape, valid_nodes.astype(bool))
        graph_predictions = reconstruct_from_n6(graph_predictions, node_voxel, roi_vol_shape)  # recovering the volume shape

        refined = graph_predictions

        # recovering sizes
        refined_expanded = np.zeros(np_pet_vol.shape, dtype=float)
        refined_expanded[roi_limits[0]:roi_limits[3], roi_limits[1]:roi_limits[4], roi_limits[2]:roi_limits[5]] = refined

        #np.save("refined.npy", refined)
        nib.save(nib.Nifti1Image(refined_expanded, nii_seg.affine), sample_seg)

