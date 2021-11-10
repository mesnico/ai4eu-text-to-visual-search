import h5py

from utils import get_model
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import os
import pickle
import re
import faiss

mirflickr100k_root = '/mnt/MIRFlickr100k'
out_features_filename = '/mnt/MIRFlickr100k/MIRf100k_tern_features.h5'
coco_features_filename = 'indexing_data/coco_test_tern_features.h5'
bs = 30

model_checkpoint = "runs/ALIGN/WITH_ICLS/tere_alignment_tern_teran_uncertainty_weighting/model_best_ndcgspice.pth.tar"

out_index_path = 'ai4eu/faiss_mirf100k/' #features.faiss'

class BottomUpFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features_path):
        # which dataset?

        # data_path = config['image-model']['data-path']
        self.feats_data_path = os.path.join(features_path, 'bu_att')
        self.box_data_path = os.path.join(features_path, 'bu_box')
        img_sizes_filename = os.path.join(features_path, '../image_sizes.pkl')
        with open(img_sizes_filename, 'rb') as f:
            self.img_sizes = pickle.load(f)

        self.file_list = os.listdir(self.feats_data_path)
        self.file_list = [os.path.splitext(fl)[0] for fl in self.file_list]

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        img_id = self.file_list[index]
        img_feat_path = os.path.join(self.feats_data_path, '{}.npz'.format(img_id))
        img_box_path = os.path.join(self.box_data_path, '{}.npy'.format(img_id))
        img_size = self.img_sizes[img_id]

        img_feat = np.load(img_feat_path)['feat']
        img_boxes = np.load(img_box_path)

        # normalize boxes
        img_boxes = img_boxes / np.tile(img_size, 2)

        img_feat = torch.Tensor(img_feat)
        img_boxes = torch.Tensor(img_boxes)

        # image = (img_feat, img_boxes)
        return img_feat, img_boxes, img_id, index

    def __len__(self):
        return len(self.file_list)


class Collate:
    def __call__(self, data):
        """Build mini-batch tensors from a list of (image, caption) tuples.
            Args:
                data: list of (image, caption) tuple.
                    - image: torch tensor of shape (3, 256, 256) or (? > 3, 2048)
                    - caption: torch tensor of shape (?); variable length.

            Returns:
                images: torch tensor of shape (batch_size, 3, 256, 256).
                targets: torch tensor of shape (batch_size, padded_length).
                lengths: list; valid length for each padded caption.
            """
        images, boxes, img_ids, indexes = zip(*data)


        # they are image features, variable length
        feat_lengths = [f.shape[0] + 1 for f in images]  # +1 because the first region feature is reserved as CLS
        feat_dim = images[0].shape[1]
        img_features = torch.zeros(len(images), max(feat_lengths), feat_dim)
        for i, img in enumerate(images):
            end = feat_lengths[i]
            img_features[i, 1:end] = img

        box_lengths = [b.shape[0] + 1 for b in boxes]  # +1 because the first region feature is reserved as CLS
        assert box_lengths == feat_lengths
        out_boxes = torch.zeros(len(boxes), max(box_lengths), 4)
        for i, box in enumerate(boxes):
            end = box_lengths[i]
            out_boxes[i, 1:end] = box

        # features = features.permute(0, 2, 1)
        return img_features, feat_lengths, out_boxes, img_ids, indexes


def extract_features():
    checkpoint = torch.load(model_checkpoint, map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
    config = checkpoint['config']

    #with open(config, 'r') as ymlfile:
    #    config = yaml.load(ymlfile)

    config['training']['word-reconstruction'] = False
    config['training']['region-reconstruction'] = False

    model = get_model(config)
    # load model state
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    features_path = os.path.join(mirflickr100k_root, 'features_36')
    dataset = BottomUpFeaturesDataset(features_path)
    collate = Collate()
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate)

    with h5py.File(out_features_filename, 'w') as f:
        feats = f.create_dataset("features", (len(data_loader.dataset), 1024), dtype=np.float32)
        dt = h5py.string_dtype(encoding='utf-8')
        img_ids = f.create_dataset('image_names', (len(data_loader.dataset), ), dtype=dt)

        for i, (images, img_length, boxes, image_names, dataset_idxs) in enumerate(tqdm.tqdm(data_loader)):
            # compute the embeddings
            with torch.no_grad():
                img_emb_aggr, cap_emb_aggr, _, _, _ = model.forward_emb(images=images, captions=None, img_len=img_length, cap_len=None, boxes=boxes)
                assert cap_emb_aggr is None
                img_emb_aggr = img_emb_aggr.cpu().numpy()

                feats[dataset_idxs, :] = img_emb_aggr
                for image_name, idx in zip(image_names, dataset_idxs):
                    img_ids[idx] = np.array(image_name.encode("utf-8"), dtype=dt)
                # for img_feat, img_id in zip(img_emb_aggr, ids):
                #     db[img_id] = img_feat

def get_urls_dict():
    tot_duplicated = 0
    mirf100k_meta_files = [f for f in os.listdir(os.path.join(mirflickr100k_root, 'images')) if
                           os.path.splitext(f)[1] == '.txt']
    ids_to_urls_dict = {}
    for meta_fname in tqdm.tqdm(mirf100k_meta_files):
        meta_fname = os.path.join(mirflickr100k_root, 'images', meta_fname)
        with open(meta_fname, 'r', encoding="utf-8") as f:
            lines = f.read().splitlines()
            matches = [re.search(r"^(.*) http://static.flickr.com/\d+/(\d+)_", txt) for txt in lines]
            mirf_names, _ = zip(*[(m.group(1), m.group(2)) for m in matches])
            mirf_names = [os.path.splitext(fn)[0] for fn in mirf_names]
            urls = [l.split(' ')[1] for l in lines] # the urls if after the first space

            duplicated_ids = [(i, k) for i, k in enumerate(mirf_names) if k in ids_to_urls_dict]
            if len(duplicated_ids) != 0:   # ids should not be duplicated!
                assert all([ids_to_urls_dict[dup_id] == urls[i] for i, dup_id in duplicated_ids])
                tot_duplicated += len(duplicated_ids)

            ids_to_urls_dict.update({k: v for k, v in zip(mirf_names, urls)})

    print('Total number of duplicated images: {}'.format(tot_duplicated))
    return ids_to_urls_dict


def build_index(img_features, code_size=32, quantization='PQ', number_of_vectors_per_cell=1024):
    # load coco features for training the index
    data = h5py.File(coco_features_filename, 'r')
    coco_img_embs = data['img_embs']
    coco_cap_embs = data['cap_embs']
    dim = coco_img_embs.shape[1]

    n_cells = (len(coco_img_embs) // 5) // number_of_vectors_per_cell
    if quantization == 'PQ':
        index = faiss.index_factory(dim, 'IVF{},PQ{}'.format(n_cells, code_size))
    elif quantization == 'SQ':
        index = faiss.index_factory(dim, 'IVF{},SQ8'.format(n_cells, code_size))

    # train the index
    print('Training the Index...')
    training_data = np.concatenate((coco_img_embs[::5], coco_cap_embs[:5000], img_features[:5000]), axis=0)
    # training_data = np.array(training_data)
    index.train(training_data)
    # add elements to the index
    print('Adding region vectors to the Index...')
    # for e, i in enumerate(tqdm.trange(0, len(img_embs), 5)):
    #     ids = np.array([e] * 36)
    #     embs = img_embs[i, 1:, :]
    #     index.add_with_ids(embs, ids)
    index.add(img_features[:])
    return index


if __name__ == '__main__':
    if not os.path.exists(out_features_filename):
        extract_features()

    image_data = h5py.File(out_features_filename, 'r')
    features = image_data['features']
    ids = image_data['image_names']

    ids_to_urls_dict = get_urls_dict()
    spurious_ids = len([i for i in ids if i not in ids_to_urls_dict])
    print('Spurious ids: {}'.format(spurious_ids))
    urls = [ids_to_urls_dict[id] for id in ids]

    # build and train the FAISS index
    print('Building and training the index...')
    index = build_index(features)

    if not os.path.exists(out_index_path):
        os.makedirs(out_index_path)

    out_index_name = os.path.join(out_index_path, 'index.faiss')
    print('Saving the index in file {}'.format(out_index_name))
    faiss.write_index(index, out_index_name)

    out_urls_name = os.path.join(out_index_path, 'urls.pkl')
    print('Saving the urls in file {}'.format(out_urls_name))
    with open(out_urls_name, 'wb') as f:
        pickle.dump(urls, f)
