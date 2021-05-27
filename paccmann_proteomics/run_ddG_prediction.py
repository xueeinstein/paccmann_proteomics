"""
Finetuning pairwise RoBERTa model for ddG prediction task
"""
import dataclasses
import os
import glob
import re
import sys
import scipy
import pickle
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Tuple
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from utils.metrics_clf import glue_compute_metrics
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import default_data_collator
from loguru import logger
from paccmann_proteomics.data.processors.seq_clf import (
    seq_clf_output_modes, seq_clf_tasks_num_labels)
from paccmann_proteomics.data.datasets.ddG_reg import ddGDataset

from ddG_predictor import RobertaForddGPrediction


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            'help':
            'Path to pretrained model or model identifier from huggingface.co/models'
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained config name or path if not the same as model_name'
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Pretrained tokenizer name or path if not the same as model_name'
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'Where do you want to store the pretrained models downloaded from AWS'
        })
    continue_from_checkpoint: bool = field(
        default=False,
        metadata={
            'help':
            'Whether to continue training from `model_name_or_path/checkpoint-<Trainer.global_step>/`'
        },
    )


def _sorted_checkpoints(model_dir,
                        checkpoint_prefix='checkpoint',
                        use_modification_time=False) -> List[str]:
    """
    TODO: refactor to paccmann_proteomics/utils
    Private method discovers model checkpoint directories within the global model_dir, 
    returns a sorted list with directory paths.

    Supports sorting by checkpoint directory modification time (use with care!) or directory suffix 
    which contains the global step at which checkpoint directory was saved.

    Args:
        model_dir ([type]): location of model with checkpoint directories
        checkpoint_prefix (str, optional): directory prefix concatenated 
            with "-<Trainer.global_step>" . Defaults to "checkpoint".
        use_mtime (bool, optional): sort by directory modification time. Defaults to False.

    Returns:
        List[str]: [description]
    
    Dependancies:
        Glob, Re
    """
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(model_dir, '{}-*'.format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_modification_time:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix),
                                   path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser(dataclass_types=(ModelArguments,
                                               DataTrainingArguments,
                                               TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        )

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
        )

    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s'
        % (training_args.local_rank, training_args.device, training_args.n_gpu,
           bool(training_args.local_rank != -1), training_args.fp16))
    logger.info('Training/evaluation parameters %s' % training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load task-specific number of labels (==1 if regression) and output modes)
    try:
        num_labels = seq_clf_tasks_num_labels[data_args.task_name]
        logger.info('number of labels: %s' % num_labels)
        output_mode = seq_clf_output_modes[data_args.task_name]
        logger.info('task output mode: %s' % output_mode)
    except KeyError:
        raise ValueError('Task not found: %s' % (data_args.task_name))

    # Load pretrained model and tokenizer
    if model_args.config_name:
        logger.info('config_name provided as: %s' % model_args.config_name)
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            cache_dir=model_args.cache_dir)

    elif model_args.model_name_or_path:
        logger.info('model_name_or_path provided as: %s' %
                    model_args.model_name_or_path)

        if model_args.continue_from_checkpoint:
            logger.info(
                'checking for the newest checkpoint directory %s/checkpoint-<Trainer.global_step>'
                % model_args.model_name_or_path)
            sorted_checkpoints = _sorted_checkpoints(
                model_args.model_name_or_path)
            logger.info('checkpoints found: %s' % sorted_checkpoints)
            if len(sorted_checkpoints) == 0:
                raise ValueError(
                    'Used --continue_from_checkpoint but no checkpoint was found in --model_name_or_path.'
                )
            else:
                model_args.model_name_or_path = sorted_checkpoints[-1]

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # use_fast=True,
    )

    # Get datasets
    dataset = ddGDataset(args=data_args, tokenizer=tokenizer,
                         cache_dir=model_args.cache_dir)

    def _create_dataloader(dataset_, shuffle=True):
        dataloader = DataLoader(
            dataset_,
            shuffle=shuffle,
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=default_data_collator,
            drop_last=training_args.dataloader_drop_last,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory)
        return dataloader

    def _create_model_opt():
        model = RobertaForddGPrediction.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        model = model.to(training_args.device)

        optimizer = AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay)

        return model, optimizer

    total_rmse = total_Rp = 0.0
    ddG_preds_labels = []
    for split_id in range(dataset.n_splits):
        model, optimizer = _create_model_opt()

        best_rmse, best_Rp = 1e6, -1.
        best_ddG_preds, ddG_labels_copy = None, None
        for epoch_id in range(int(training_args.num_train_epochs)):
            dataset.select_kfold(split_id, 'train')
            dataloader = _create_dataloader(dataset, shuffle=True)
            model.train()

            for batch in dataloader:
                batch_ = {k: v.to(training_args.device)
                          for k, v in batch.items()}
                loss, preds = model(**batch_)
                loss.backward()
                optimizer.step()
                model.zero_grad()

            dataset.select_kfold(split_id, 'test')
            dataloader = _create_dataloader(dataset, shuffle=False)
            model.eval()

            ddG_labels, ddG_preds = [], []
            for batch in dataloader:
                ddG_label = batch['labels'].numpy()
                batch_ = {k: v.to(training_args.device)
                          for k, v in batch.items()}
                _, ddG_pred = model(**batch_)
                ddG_pred = ddG_pred.cpu().detach().numpy()

                ddG_labels.extend(list(ddG_label))
                ddG_preds.extend(list(ddG_pred))

            ddG_labels = dataset.scale_back_ddG(np.array(ddG_labels))
            ddG_preds = dataset.scale_back_ddG(np.array(ddG_preds))
            rmse = np.sqrt(((ddG_labels - ddG_preds) ** 2).mean())
            Rp = scipy.stats.pearsonr(ddG_preds, ddG_labels)[0]
            print('====== Rp:', Rp, ' RMSE:', rmse)
            if np.isnan(Rp):
                # Overfit! ddG_pred becomes a constant
                break
            if Rp > best_Rp:
                best_Rp, best_rmse = Rp, rmse
                best_ddG_preds = ddG_preds.copy()
                ddG_labels_copy = ddG_labels.copy()

        print('Fold: {}, Rp: {:.3f}, RMSE: {:.3f}'.format(
            split_id, best_Rp, best_rmse))
        total_rmse += best_rmse
        total_Rp += best_Rp
        ddG_preds_labels.append((best_ddG_preds, ddG_labels_copy))

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    with open(os.path.join(training_args.output_dir, 'metric.txt'), 'w') as f:
        avg_rmse = total_rmse / dataset.n_splits
        avg_Rp = total_Rp / dataset.n_splits
        metric = 'Rp: {:.3f}, RMSE: {:.3f}'.format(avg_Rp, avg_rmse)
        f.write(metric)
        print(metric)

    with open(os.path.join(training_args.output_dir, 'preds_labels.pkl'), 'wb') as f:
        pickle.dump(ddG_preds_labels, f)


if __name__ == '__main__':
    main()
