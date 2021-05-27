import os
import time
import pandas as pd
from loguru import logger
from typing import Optional
from filelock import FileLock
from sklearn.model_selection import KFold

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from transformers.data.datasets import GlueDataTrainingArguments
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features

from ..processors.seq_clf import seq_clf_output_modes


def apply_kfold(n_splits, seed, data_list):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kf_splits = []
    for train_ids, test_ids in kf.split(data_list):
        kf_splits.append((train_ids, test_ids))

    return kf_splits


def load_ddg_seq_file(id2seq_file, ddg_file, with_affinity_label):
    id2seq, seqs = dict(), []
    with open(id2seq_file, 'r') as f:
        seq_id = 0
        for line in f:
            line = line.strip().split(' ')
            id2seq[line[0]] = seq_id
            seqs.append(line[1])
            seq_id += 1

    df = pd.read_csv(ddg_file, sep=' ', header=None)
    c = len(df.columns)

    if with_affinity_label:
        assert c == 4 + 3
        dG_w_max, dG_w_min = max(df[4]), min(df[4])
        dG_m_max, dG_m_min = max(df[5]), min(df[5])
        ddG_max, ddG_min = max(df[6]), min(df[6])
    else:
        # the last column is always ddG
        ddG_max, ddG_min = max(df[c-1]), min(df[c-1])

    raw_ddg_data = []
    for i in range(len(df)):
        wild_seq1_id = id2seq[df.iloc[i][0]]
        wild_seq2_id = id2seq[df.iloc[i][1]]
        mut_seq1_id = id2seq[df.iloc[i][2]]
        mut_seq2_id = id2seq[df.iloc[i][3]]
        data = [wild_seq1_id, wild_seq2_id, mut_seq1_id, mut_seq2_id]
        if with_affinity_label:
            data.append((df.iloc[i][6] - ddG_min) / (ddG_max - ddG_min))
            data.append((df.iloc[i][4] - dG_w_min) / (dG_w_max - dG_w_min))
            data.append((df.iloc[i][5] - dG_m_min) / (dG_m_max - dG_m_min))
        else:
            data.append((df.iloc[i][c-1] - ddG_min) / (ddG_max - ddG_min))

        raw_ddg_data.append(data)

    return id2seq, seqs, raw_ddg_data, ddG_max, ddG_min


class ddGDataset(Dataset):
    def __init__(self,
                 args: GlueDataTrainingArguments,
                 tokenizer: PreTrainedTokenizer,
                 n_splits: int = 10,
                 kfold_seed: int = 13,
                 with_affinity_label: bool = False,
                 cache_dir: Optional[str] = None):

        if with_affinity_label:
            # Currently, don't support dG label
            raise NotImplementedError

        self.args = args
        self.output_mode = seq_clf_output_modes[args.task_name]

        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            'cached_{}_{}_{}'.format(
                tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )

        id2seq_file = os.path.join(args.data_dir, '%s.seq.txt' % args.task_name)
        ddg_file = os.path.join(args.data_dir, '%s.ddg.txt' % args.task_name)
        self.id2seq, self.seqs, self.raw_ddg_data, self.ddG_max, \
            self.ddG_min = load_ddg_seq_file(
                id2seq_file, ddg_file, with_affinity_label)

        # Process to features
        lock_path = cached_features_file + '.lock'
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f'Loading features from cached file {cached_features_file} [took %.3f s]' % (time.time() - start)
                )
            else:
                logger.info(f'Creating features from dataset file at {args.data_dir}')

                self.features = self._process_data(tokenizer)
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    'Saving features into cached file %s [took %.3f s]' % (cached_features_file, time.time() - start)
                )

        self.n_splits = n_splits
        self.kf_splits = apply_kfold(n_splits, kfold_seed, self.features)
        self.select_kfold(0, 'train')

    def select_kfold(self, k, ds_type):
        assert k < len(self.kf_splits)
        self.kfold_id = k
        self.train_ids, self.test_ids = self.kf_splits[k]

        if ds_type == 'train':
            self.ids = self.train_ids
        elif ds_type == 'test':
            self.ids = self.test_ids
        else:
            raise ValueError

    def scale_back_ddG(self, ddG):
        return ddG * (self.ddG_max - self.ddG_min) + self.ddG_min

    def _process_data(self, tokenizer):
        examples = []
        for i, data in enumerate(self.raw_ddg_data):
            wild_seq1_id, wild_seq2_id, mut_seq1_id, mut_seq2_id, ddG = data[:5]

            examples.append(InputExample(
                guid='wild-%d' % i,
                text_a=self.seqs[wild_seq1_id],
                text_b=self.seqs[wild_seq2_id],
                label=ddG))

            examples.append(InputExample(
                guid='mut-%d' % i,
                text_a=self.seqs[mut_seq1_id],
                text_b=self.seqs[mut_seq2_id],
                label=ddG))

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=self.args.max_seq_length,
            label_list=['ddG'],
            output_mode=self.output_mode)

        features_ = []
        for i in range(len(features) // 2):
            wild_feat = features[i * 2]
            mut_feat = features[i * 2 + 1]

            input_ids = [wild_feat.input_ids, mut_feat.input_ids]
            attn_mask = [wild_feat.attention_mask, mut_feat.attention_mask]
            label = wild_feat.label
            if wild_feat.token_type_ids is None:
                token_type_ids = None
            else:
                token_type_ids = [wild_feat.token_type_ids,
                                  mut_feat.token_type_ids]

            features_.append(InputFeatures(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
                label=label))

        return features_

    def __getitem__(self, idx):
        idx = idx % len(self.ids)
        return self.features[self.ids[idx]]

    def __len__(self):
        return len(self.ids)
