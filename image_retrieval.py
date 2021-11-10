import torch
import faiss
import os
import pickle
import numpy as np

from transformers import BertTokenizer

from utils import get_model

class ImageRetrieval:
    def __init__(self, faiss_index_path, tern_checkpoint_filename):
        # Load index and image urls
        self.index = faiss.read_index(os.path.join(faiss_index_path, 'index.faiss'))
        with open(os.path.join(faiss_index_path, 'urls.pkl'), 'rb') as f:
            self.urls = pickle.load(f)

        # Initialize model
        checkpoint = torch.load(tern_checkpoint_filename,
                                map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device(
                                    'cuda'))
        config = checkpoint['config']

        config['training']['word-reconstruction'] = False
        config['training']['region-reconstruction'] = False
        model = get_model(config)
        # load model state
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        self.model = model

        # Initialize BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])

    def encode_query(self, query):
        # given a caption query, compute its 1024-dimensional representation
        query_length = len(self.tokenizer.tokenize(query)) + 2  # + 2 in order to account for begin and end tokens
        captions_ids = torch.LongTensor(self.tokenizer.encode(query, max_length=query_length, pad_to_max_length=True))
        captions_ids = captions_ids.unsqueeze(0)  # B x len (B = 1)
        captions_ids = captions_ids.cuda() if torch.cuda.is_available() else captions_ids

        # forward the model
        with torch.no_grad():
            img_emb_aggr, cap_emb_aggr, _, _, _ = self.model.forward_emb(images=None, captions=captions_ids,
                                                                         img_len=None,
                                                                         cap_len=[query_length], boxes=None)
        assert img_emb_aggr is None
        cap_emb_aggr = cap_emb_aggr.cpu().squeeze(0).numpy()
        return cap_emb_aggr

    def search(self, query, K=100):
        # get the query feature
        query_emb = self.encode_query(query)
        # do the search
        D, I = self.index.search(np.expand_dims(query_emb, 0), K)
        D = D.flatten()
        I = I.flatten()
        most_relevant_img_urls = [self.urls[i] for i in I]
        scores = 1 / (D + 1)    # convert euclidean distance into a similarity score
        return most_relevant_img_urls, scores


if __name__ == '__main__':
    faiss_index_path = 'faiss_mirf100k/'
    tern_checkpoint = "tern_checkpoints/model_tern_teran_uncertainty_best_ndcgspice.pth.tar"
    ir = ImageRetrieval(faiss_index_path, tern_checkpoint)

    while(1):
        query = input("Enter a textual query: ")
        img_urls, similarities = ir.search(query, 10)
        print(img_urls[:20])