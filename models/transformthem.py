import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer

from models.loss import ContrastiveLoss, PermInvMatchingLoss, AlignmentContrastiveLoss
from models.text import EncoderText
from models.visual import EncoderImage

from .utils import l2norm, PositionalEncodingImageBoxes, PositionalEncodingText, Aggregator, generate_square_subsequent_mask
# from nltk.corpus import stopwords, words as nltk_words


class JointTextImageTransformerEncoder(nn.Module):
    """
    This is a bert caption encoder - transformer image encoder (using bottomup features).
    If process the encoder outputs through a transformer, like VilBERT and outputs two different graph embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.txt_enc = EncoderText(config)
        self.txt_mode = config['text-model']['name']

        visual_feat_dim = config['image-model']['feat-dim']
        caption_feat_dim = config['text-model']['word-dim']
        dropout = config['model']['dropout']
        layers = config['model']['layers']
        embed_size = config['model']['embed-size']
        self.order_embeddings = config['training']['measure'] == 'order'
        self.loss_type = config['training']['loss-type']
        self.img_enc = EncoderImage(config)

        self.img_proj = nn.Linear(visual_feat_dim, embed_size)
        self.cap_proj = nn.Linear(caption_feat_dim, embed_size)
        self.embed_size = embed_size
        self.shared_transformer = config['model']['shared-transformer']

        dim_feedforward = config['model']['feedforward-dim'] if 'feedforward-dim' in config['model'] else 2048  # for backward compatibility

        if 'crossattention' in self.loss_type:
            self.cross_attention_aggregation = CrossAttentionAggregation(d_model=config['model']['embed-size'], feedforward_dim=2048)

        transformer_layer_1 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder_1 = nn.TransformerEncoder(transformer_layer_1,
                                                           num_layers=layers)
        if not self.shared_transformer:
            transformer_layer_2 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                             dim_feedforward=dim_feedforward,
                                                             dropout=dropout, activation='relu')
            self.transformer_encoder_2 = nn.TransformerEncoder(transformer_layer_2,
                                                               num_layers=layers)
        self.text_aggregation = Aggregator(embed_size, aggregation_type=config['model']['text-aggregation'])
        self.image_aggregation = Aggregator(embed_size, aggregation_type=config['model']['image-aggregation'])
        self.text_aggregation_type = config['model']['text-aggregation']
        self.img_aggregation_type = config['model']['image-aggregation']

    def forward(self, features, captions, feat_len, cap_len, boxes):
        # process captions by using bert
        if captions is not None:
            full_cap_emb_aggr, c_emb = self.txt_enc(captions, cap_len)     # B x S x cap_dim

        # process image regions using a two-layer transformer
        if features is not None:
            full_img_emb_aggr, i_emb = self.img_enc(features, feat_len, boxes)     # B x S x vis_dim
        # i_emb = i_emb.permute(1, 0, 2)                             # B x S x vis_dim

        bs = features.shape[0] if features is not None else captions.shape[0]

        # if False:
        #     # concatenate the embeddings together
        #     max_summed_lengths = max([x + y for x, y in zip(feat_len, cap_len)])
        #     i_c_emb = torch.zeros(bs, max_summed_lengths, self.embed_size)
        #     i_c_emb = i_c_emb.to(features.device)
        #     mask = torch.zeros(bs, max_summed_lengths).bool()
        #     mask = mask.to(features.device)
        #     for i_c, m, i, c, i_len, c_len in zip(i_c_emb, mask, i_emb, c_emb, feat_len, cap_len):
        #         i_c[:c_len] = c[:c_len]
        #         i_c[c_len:c_len + i_len] = i[:i_len]
        #         m[c_len + i_len:] = True
        #
        #     i_c_emb = i_c_emb.permute(1, 0, 2)      # S_vis + S_txt x B x dim
        #     out = self.transformer_encoder(i_c_emb, src_key_padding_mask=mask)      # S_vis + S_txt x B x dim
        #
        #     full_cap_emb = out[0, :, :]
        #     I = torch.LongTensor(cap_len).view(1, -1, 1)
        #     I = I.expand(1, bs, self.embed_size).to(features.device)
        #     full_img_emb = torch.gather(out, dim=0, index=I).squeeze(0)
        # else:

        # forward the captions
        if self.text_aggregation_type is not None and captions is not None:
            if self.txt_mode != 'gru' and self.txt_mode != 'lstm':
                c_emb_proj = self.cap_proj(c_emb)
                c_emb_proj = c_emb_proj.permute(1, 0, 2)

            mask = torch.zeros(bs, max(cap_len)).bool()
            mask = mask.to(captions.device)
            for m, c_len in zip(mask, cap_len):
                m[c_len:] = True
            full_cap_emb = self.transformer_encoder_1(c_emb_proj, src_key_padding_mask=mask)  # S_txt x B x dim
            full_cap_emb_aggr = self.text_aggregation(full_cap_emb, cap_len, mask)
        # else use the embedding output by the txt model
        else:
            full_cap_emb = None

        # forward the regions
        if self.img_aggregation_type is not None and features is not None:
            i_emb_proj = self.img_proj(i_emb)
            i_emb_proj = i_emb_proj.permute(1, 0, 2)

            mask = torch.zeros(bs, max(feat_len)).bool()
            mask = mask.to(features.device)
            for m, v_len in zip(mask, feat_len):
                m[v_len:] = True
            if self.shared_transformer:
                full_img_emb = self.transformer_encoder_1(i_emb_proj, src_key_padding_mask=mask)  # S_txt x B x dim
            else:
                full_img_emb = self.transformer_encoder_2(i_emb_proj, src_key_padding_mask=mask)  # S_txt x B x dim
            full_img_emb_aggr = self.image_aggregation(full_img_emb, feat_len, mask)
        else:
            full_img_emb = None

        # normalize even every vector of the set
        full_img_emb = F.normalize(full_img_emb, p=2, dim=2) if full_img_emb is not None else None
        full_cap_emb = F.normalize(full_cap_emb, p=2, dim=2) if full_cap_emb is not None else None

        if 'crossattention' in self.loss_type:
            img_emb_set = full_img_emb.permute(1, 0, 2)
            cap_emb_seq = full_cap_emb.permute(1, 0, 2)
            full_img_emb_aggr, full_cap_emb_aggr = self.cross_attention_aggregation(img_emb_set, cap_emb_seq, feat_len,
                                                                                cap_len)
            # re-inject I-CLS tokens and T-CLS tokens
            # full_img_emb[0, :, :] = full_img_emb_aggr
            # full_cap_emb[0, :, :] = full_cap_emb_aggr

        full_cap_emb_aggr = l2norm(full_cap_emb_aggr) if captions is not None else None
        full_img_emb_aggr = l2norm(full_img_emb_aggr) if features is not None else None

        if self.order_embeddings:
            full_cap_emb_aggr = torch.abs(full_cap_emb_aggr)
            full_img_emb_aggr = torch.abs(full_img_emb_aggr)

        # output word and region embeddings at different stages of the architecture
        img_embs = [i_emb_proj, full_img_emb] if features is not None else None
        cap_embs = [c_emb_proj, full_cap_emb] if captions is not None else None
        return full_img_emb_aggr, full_cap_emb_aggr, img_embs, cap_embs


class TransformThem(torch.nn.Module):
    """
    rkiros/uvs model
    """

    def __init__(self, config):
        # tutorials/09 - Image Captioning
        # Build Models
        super().__init__()
        self.img_txt_enc = JointTextImageTransformerEncoder(config)
        if torch.cuda.is_available():
            self.img_txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer

        self.angular_multiplier = config['training']['angular-multiplier'] if 'angular-multiplier' in config['training'] else 0
        loss_type = config['training']['loss-type']
        force_alignment = config['training']['force-alignment'] if 'force-alignment' in config['training'] else False
        domain_adaptation = config['training']['domain-adaptation'] if 'domain-adaptation' in config['training'] else None
        if domain_adaptation is not None and domain_adaptation != 'adv':
            self.adaptation_criterion = Adaptation(domain_adaptation)
        else:
            self.adaptation_criterion = None

        if 'alignment' in loss_type:
            self.alignment_criterion = AlignmentContrastiveLoss(margin=config['training']['margin'],
                                                                measure=config['training']['measure'],
                                                                max_violation=config['training']['max-violation'], aggregation=config['training']['alignment-mode'], force_alignment=force_alignment)
        if 'matching' or 'crossattention' in loss_type:
            self.matching_criterion = ContrastiveLoss(margin=config['training']['margin'],
                                                      measure=config['training']['measure'],
                                                      max_violation=config['training']['max-violation'])
            if self.angular_multiplier > 0:
                self.angular_loss = AngularLoss(max_violation=config['training']['max-violation'])

        self.Eiters = 0
        self.config = config
        self.region_reconstruction = config['training']['region-reconstruction']
        self.word_reconstruction = config['training']['word-reconstruction']

        if 'exclude-stopwords' in config['model'] and config['model']['exclude-stopwords']:
            self.en_stops = set(stopwords.words('english'))
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
        else:
            self.tokenizer = None

        if 'alignment' in config['training']['loss-type'] and 'matching' in config['training']['loss-type']:
            self.s_tern = nn.Parameter(-2.3 * torch.ones(1))
            self.s_teran = nn.Parameter(-2.3 * torch.ones(1))

        if self.region_reconstruction:
            # Visual decoder
            self.common_space_to_region = nn.Linear(config['model']['embed-size'], config['image-model']['feat-dim'])
            visual_decoder_layer = nn.TransformerDecoderLayer(d_model=config['image-model']['feat-dim'], nhead=4,
                                                              dim_feedforward=256,
                                                              dropout=0.1, activation='relu')
            self.transformer_visual_decoder = nn.TransformerDecoder(visual_decoder_layer,
                                                                    num_layers=1)
            self.decode_visual_sequence = nn.Linear(config['image-model']['feat-dim'], config['image-model']['feat-dim'] + 4)
            self.regress = PermInvMatchingLoss()

            self.pos_encoding_image = PositionalEncodingImageBoxes(config['image-model']['feat-dim'])

        if self.word_reconstruction:
            # self.embed = nn.Embedding(self.img_txt_enc.txt_enc.vocab_size, config['text-model']['word-dim'])
            # Caption decoder
            self.common_space_to_word = nn.Linear(config['model']['embed-size'], config['text-model']['word-dim'])
            caption_decoder_layer = nn.TransformerDecoderLayer(d_model=config['text-model']['word-dim'], nhead=4,
                                                               dim_feedforward=2048,
                                                               dropout=0.1, activation='relu')
            self.transformer_caption_decoder = nn.TransformerDecoder(caption_decoder_layer,
                                                                     num_layers=1)
            self.word_fc = nn.Linear(config['text-model']['word-dim'], self.img_txt_enc.txt_enc.vocab_size)
            self.word_classifier = nn.CrossEntropyLoss(ignore_index=0)

            self.pos_encoding_text = PositionalEncodingText(config['text-model']['word-dim'])

    # def state_dict(self):
    #     state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
    #     return state_dict
    #
    # def load_state_dict(self, state_dict):
    #     self.img_enc.load_state_dict(state_dict[0])
    #     self.txt_enc.load_state_dict(state_dict[1])
    #
    # def train_start(self):
    #     """switch to train mode
    #     """
    #     self.img_enc.train()
    #     self.txt_enc.train()
    #
    # def val_start(self):
    #     """switch to evaluate mode
    #     """
    #     self.img_enc.eval()
    #     self.txt_enc.eval()

    def forward_emb(self, images, captions, img_len, cap_len, boxes, all_layers_output=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda() if images is not None else None
            captions = captions.cuda() if captions is not None else None
            boxes = boxes.cuda() if boxes is not None else None

        # Forward
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats = self.img_txt_enc(images, captions, img_len, cap_len, boxes)

        if self.tokenizer is not None and cap_feats is not None:
            # remove stopwords
            # keep only word indexes that are not stopwords
            good_word_indexes = [[i for i, (tok, w) in enumerate(zip(self.tokenizer.convert_ids_to_tokens(ids), ids)) if
                                  tok not in self.en_stops or w == 0] for ids in captions]  # keeps the padding
            cap_len = [len(w) - (cap_feats[-1].shape[0] - orig_len) for w, orig_len in zip(good_word_indexes, cap_len)]
            min_cut_len = min([len(w) for w in good_word_indexes])
            good_word_indexes = [words[:min_cut_len] for words in good_word_indexes]
            good_word_indexes = torch.LongTensor(good_word_indexes).to(cap_feats.device) # B x S
            good_word_indexes = good_word_indexes.t().unsqueeze(2).expand(-1, -1, cap_feats.shape[2]) # S x B x dim
            cap_feats[-1] = cap_feats[-1].gather(dim=0, index=good_word_indexes)

        if not all_layers_output:
            # return only the output from the last layer
            img_feats = img_feats[-1] if img_feats is not None else None
            cap_feats = cap_feats[-1] if cap_feats is not None else None
        return img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, cap_len

    # def get_parameters(self):
    #     lr_multiplier = 1.0 if self.config['text-model']['fine-tune'] else 0.0
    #
    #     ret = []
    #     params = list(self.img_txt_enc.img_enc.parameters())
    #     params += list(self.img_txt_enc.img_proj.parameters())
    #     params += list(self.img_txt_enc.cap_proj.parameters())
    #     params += list(self.img_txt_enc.transformer_encoder_1.parameters())
    #
    #     params += list(self.img_txt_enc.image_aggregation.parameters())
    #     params += list(self.img_txt_enc.text_aggregation.parameters())
    #
    #     if not self.config['model']['shared-transformer']:
    #         params += list(self.img_txt_enc.transformer_encoder_2.parameters())
    #
    #     if self.region_reconstruction:
    #         params += list(self.common_space_to_region.parameters())
    #         params += list(self.transformer_visual_decoder.parameters())
    #         params += list(self.decode_visual_sequence.parameters())
    #         params += list(self.pos_encoding_image.parameters())
    #     if self.word_reconstruction:
    #         # params += list(self.embed.parameters())
    #         params += list(self.common_space_to_word.parameters())
    #         params += list(self.transformer_caption_decoder.parameters())
    #         params += list(self.word_fc.parameters())
    #         params += list(self.pos_encoding_text.parameters())
    #
    #     if 'alignment' in self.config['training']['loss-type'] and 'matching' in self.config['training']['loss-type']:
    #         params += list(self.s_tern)
    #         params += list(self.s_teran)
    #
    #     ret.append(params)
    #
    #     ret.append(list(self.img_txt_enc.txt_enc.parameters()))
    #
    #     return ret, lr_multiplier

    def forward_loss(self, img_emb, cap_emb, img_emb_set, cap_emb_seq, img_lengths, cap_lengths, epoch):
        """Compute the loss given pairs of image and caption embeddings
        """
        # bs = img_emb.shape[0]
        losses = {}

        if  'matching' in self.config['training']['loss-type']:
            matching_loss = self.matching_criterion(img_emb, cap_emb)
            if self.angular_multiplier > 0:
                ids = list(range(img_emb.shape[0]))
                angular_loss = self.angular_loss(img_emb, cap_emb, None, ids)
                if epoch > 20:
                    alpha = 0
                else:
                    alpha = self.angular_multiplier * (0.1 ** (epoch // 5))
                angular_loss = alpha * angular_loss
                self.logger.update('angular_loss', angular_loss.item(), img_emb.size(0))
            losses.update({'matching-loss': matching_loss})
            self.logger.update('matching_loss', matching_loss.item(), img_emb.size(0))

        if 'alignment' in self.config['training']['loss-type']:
            img_emb_set_last_layer = img_emb_set[-1].permute(1, 0, 2)
            cap_emb_seq_last_layer = cap_emb_seq[-1].permute(1, 0, 2)
            alignment_loss, force_loss = self.alignment_criterion(img_emb_set_last_layer, cap_emb_seq_last_layer, img_lengths, cap_lengths)
            losses.update({'alignment-loss': alignment_loss})
            self.logger.update('alignment_loss', alignment_loss.item(), img_emb_set_last_layer.size(0))
            if force_loss is not None:
                losses.update({'force-alignment-loss': force_loss})
                self.logger.update('force_alignment_loss', force_loss.item(), img_emb_set_last_layer.size(0))
            if self.adaptation_criterion is not None:
                # compute domain adaptation loss over the regions and words distributions, possibly at different layers
                adaptation_loss = self.adaptation_criterion(img_emb_set_last_layer, cap_emb_seq_last_layer, img_lengths, cap_lengths)
                losses.update({'adaptation-loss': adaptation_loss})
                self.logger.update('adaptation_loss', adaptation_loss.item(), img_emb_set_last_layer.size(0))

        if 'crossattention' in self.config['training']['loss-type']:
            matching_loss = self.matching_criterion(img_emb, cap_emb)
            losses.update({'cross-attention-loss': matching_loss})
            self.logger.update('cross_attention_loss', matching_loss.item(), img_emb.size(0))

        # self.logger.update('Le', matching_loss.item() + alignment_loss.item(), img_emb.size(0) if img_emb is not None else img_emb_set.size(1))
        return losses

    def caption_reconstruction_loss(self, img_feats, img_lengths, captions, cap_lengths, wembeddings):
        bs = captions.shape[0]
        captions = captions.to(img_feats.device)  # B x seq_len
        if wembeddings is not None:
            cap_words_emb = wembeddings.to(
                img_feats.device)  # self.img_txt_enc.txt_enc.word_embeddings(captions)  # B x seq_len x dim
        else:
            cap_words_emb = self.embed(captions)

        # reconstruct the words given the image feature, using a transformer decoder
        # sos_token = torch.zeros(1, bs, cap_words_emb.shape[2]).to(img_emb.device)
        # NOTE: there is already a sos token in the sequence
        tgt_sequence = cap_words_emb.permute(1, 0, 2)
        # tgt_sequence[0] = torch.zeros(bs, tgt_sequence.shape[2])
        tgt_sequence = self.pos_encoding_text(tgt_sequence)
        memory = self.common_space_to_word(img_feats) # self.common_space_to_word(img_feats.unsqueeze(1)).permute(1, 0, 2)  # is seen like a sequence of a single element

        # construct the target mask
        max_cap_len = max(cap_lengths)
        tgt_padding_mask = torch.zeros(bs, max_cap_len).bool()
        for e, l in zip(tgt_padding_mask, cap_lengths):
            e[l:] = True
        tgt_padding_mask = tgt_padding_mask.to(img_feats.device)

        # construct the source mask
        max_img_len = max(img_lengths)
        src_padding_mask = torch.zeros(bs, max_img_len).bool()
        for e, l in zip(src_padding_mask, img_lengths):
            e[l:] = True
        src_padding_mask = src_padding_mask.to(img_feats.device)

        tgt_mask = generate_square_subsequent_mask(tgt_sequence.shape[0]).to(tgt_sequence.device)
        reconstructed_sequence = self.transformer_caption_decoder(tgt_sequence, memory, memory_key_padding_mask=src_padding_mask,
                                                                  tgt_key_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)
        # exclude the last token (this is not mapped to any input token)
        reconstructed_sequence = reconstructed_sequence[:-1]
        # mask out the variable len features
        # reconstructed_sequence = reconstructed_sequence.masked_fill_(tgt_mask[:, 1:].unsqueeze(2), 0)
        out = self.word_fc(reconstructed_sequence)
        out = out.permute(1, 0, 2).contiguous()
        # captions = captions.to(img_emb.device)
        word_recons_loss = self.word_classifier(out.view(-1, self.img_txt_enc.txt_enc.vocab_size),
                                                captions[:, 1:].contiguous().view(-1))
        # word_recons_loss *= 10
        # visual_recons_loss *= 90

        self.logger.update('Le/word_recons', word_recons_loss.item(), img_feats.size(0))
        return word_recons_loss

    def region_reconstruction_loss(self, cap_feats, cap_lengths, img_feats, img_lengths, boxes):
        bs = cap_feats.shape[0]
        img_spatial_emb = img_feats.to(cap_feats.device)   # B x seq_len x dim
        boxes = boxes.to(cap_feats.device)

        # reconstruct visual features given the caption feature, using a transformer decoder
        # img_spatial_emb = img_spatial_emb.view(bs, img_spatial_emb.shape[1], -1)    # B x C x (W*H)
        sos_visual_token = torch.ones(1, bs, img_spatial_emb.shape[2]).to(cap_feats.device)  # 1 x B x dim
        tgt_visual_sequence = self.pos_encoding_image(img_spatial_emb.permute(1, 0, 2), boxes=boxes)
        tgt_visual_sequence = torch.cat([sos_visual_token, tgt_visual_sequence], dim=0)    # S+1 x B x dim

        # construct the target mask
        max_img_len = max(img_lengths) + 1
        tgt_padding_mask = torch.zeros(bs, max_img_len).bool()
        for e, l in zip(tgt_padding_mask, img_lengths):
            e[l+1:] = True
        tgt_padding_mask = tgt_padding_mask.to(cap_feats.device)

        # construct the source mask
        max_cap_len = max(cap_lengths)
        src_padding_mask = torch.zeros(bs, max_cap_len).bool()
        for e, l in zip(src_padding_mask, cap_lengths):
            e[l:] = True
        src_padding_mask = src_padding_mask.to(img_feats.device)

        memory = self.common_space_to_region(cap_feats)  # is seen like a sequence of a single element
        tgt_mask = generate_square_subsequent_mask(tgt_visual_sequence.shape[0]).to(tgt_visual_sequence.device)
        reconstructed_visual_sequence = self.transformer_visual_decoder(tgt_visual_sequence,
                                                                        memory,
                                                                        memory_key_padding_mask=src_padding_mask,
                                                                        tgt_key_padding_mask=tgt_padding_mask,
                                                                        tgt_mask=tgt_mask)
        reconstructed_visual_sequence = self.decode_visual_sequence(reconstructed_visual_sequence)
        # exclude the last token (the eos token)
        reconstructed_visual_sequence = reconstructed_visual_sequence[:-1]
        reconstructed_visual_sequence = reconstructed_visual_sequence.permute(1, 0, 2).contiguous()
        target = torch.cat([img_spatial_emb, boxes], dim=2)
        visual_recons_loss = self.regress(target, reconstructed_visual_sequence)   # standard regression does not work. Permutation invariance required

        self.logger.update('Le/region_recons', visual_recons_loss.item(), cap_feats.size(0))
        return visual_recons_loss

    def forward(self, images, targets, img_lengths, cap_lengths, boxes=None, ids=None, epoch=0, return_feats=False):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets
            wembeddings = self.img_txt_enc.txt_enc.word_embeddings(captions.cuda() if torch.cuda.is_available() else captions)

        # compute the embeddings
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, cap_lengths = self.forward_emb(images, text, img_lengths, cap_lengths, boxes, all_layers_output=True)
        # NOTE: img_feats and cap_feats are S x B x dim

        loss_dict = self.forward_loss(img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_lengths, cap_lengths, epoch)

        # loss_dict = {'retrieval_loss': retrieval_loss}
        if self.word_reconstruction:
            caption_recons_loss = self.caption_reconstruction_loss(img_feats, img_lengths, captions=captions, cap_lengths=cap_lengths, wembeddings=wembeddings)
            loss_dict.update({'caption_recons': caption_recons_loss})
        if self.region_reconstruction:
            region_recons_loss = self.region_reconstruction_loss(cap_feats, cap_lengths, img_feats=images, img_lengths=img_lengths, boxes=boxes)
            loss_dict.update({'region_recons': region_recons_loss})

        # aggregate losses using uncertainty
        if 'matching-loss' in loss_dict and 'alignment-loss' in loss_dict:
            loss = torch.exp(-self.s_tern) * loss_dict['matching-loss'] + torch.exp(-self.s_teran) * loss_dict['alignment-loss'] + \
                       (self.s_tern + self.s_teran)
            loss *= 0.5
        else:
            loss = sum(l for l in loss_dict.values())

        if return_feats:
            return loss, (img_feats, cap_feats, img_lengths, cap_lengths)
        return loss
