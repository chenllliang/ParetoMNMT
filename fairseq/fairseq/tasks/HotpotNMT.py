# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import time

import torch
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    iterators,
)
from fairseq.data.multilingual.hotpot_multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction
from fairseq.optim.amp_optimizer import AMPOptimizer


###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


###


logger = logging.getLogger(__name__)


@register_task("HotPotNMT")
class HotPotNMT(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--dirs-keep', default=None)

        parser.add_argument('--use-task-drop',default=False,action='store_true')

        parser.add_argument('--log-gradient-var',default=False,action='store_true')



        parser.add_argument('--conflict-drop-task',default=None)
        parser.add_argument('--conflict-drop-layers',default=None)
        parser.add_argument('--conflict-drop-prob',type=float,default=0.5)


        
        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
        self.langs = langs
        self.dicts = dicts
        self.training = training

        self.use_task_drop = args.use_task_drop
        self.conflict_drop_prob = args.conflict_drop_prob
        self.log_gradient_var = args.log_gradient_var

        self.dev_nll_losses_each_task = []

        if args.conflict_drop_task is not None:
            self.conflict_drop_task = args.conflict_drop_task.split(",")
        else:
            self.conflict_drop_task = None
        

        if args.conflict_drop_layers is not None:
            self.conflict_drop_layers = eval(args.conflict_drop_layers)
        else:
            self.conflict_drop_layers = None

        if args.dirs_keep!=None:
            self.dirs_keep = args.dirs_keep.split(",")
        else:
            self.dirs_keep = args.dirs_keep
        

        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )
        self.logged_step=set()

    def check_dicts(self, dicts, source_langs, target_langs):
        if self.args.source_dict is not None or self.args.target_dict is not None:
            # no need to check whether the source side and target side are sharing dictionaries
            return
        src_dict = dicts[source_langs[0]]
        tgt_dict = dicts[target_langs[0]]
        for src_lang in source_langs:
            assert (
                src_dict == dicts[src_lang]
            ), "Diffrent dictionary are specified for different source languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all source languages"
        for tgt_lang in target_langs:
            assert (
                tgt_dict == dicts[tgt_lang]
            ), "Diffrent dictionary are specified for different target languages; "
            "TranslationMultiSimpleEpochTask only supports one shared dictionary across all target languages"

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)

    def has_sharded_data(self, split):
        return self.data_manager.has_sharded_data(split)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split):
                if self.args.virtual_epoch_size is not None:
                    if dataset.load_next_shard:
                        shard_epoch = dataset.shard_epoch
                    else:
                        # no need to load next shard so skip loading
                        # also this avoid always loading from beginning of the data
                        return
                else:
                    shard_epoch = epoch
        else:
            # estimate the shard epoch from virtual data size and virtual epoch size
            shard_epoch = self.data_manager.estimate_global_pass_epoch(epoch)
        logger.info(f"loading data for {split} epoch={epoch}/{shard_epoch}")
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        if split in self.datasets:
            del self.datasets[split]
            logger.info("old dataset deleted manually")
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        self.datasets[split] = self.data_manager.load_dataset(
            split,
            self.training,
            epoch=epoch,
            combine=combine,
            shard_epoch=shard_epoch,
            **kwargs,
        )



    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=self.args.source_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=self.args.target_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        return dataset

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_model(self, args, from_checkpoint=False):
        return super().build_model(args, from_checkpoint)
    
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

    #     """
    #     Do forward and backward, and return the loss as computed by *criterion*
    #     for the given *model* and *sample*.

    #     Args:
    #         sample (dict): the mini-batch. The format is defined by the
    #             :class:`~fairseq.data.FairseqDataset`.
    #         model (~fairseq.models.BaseFairseqModel): the model
    #         criterion (~fairseq.criterions.FairseqCriterion): the criterion
    #         optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
    #         update_num (int): the current update
    #         ignore_grad (bool): multiply loss by 0 if this is set to True

    #     Returns:
    #         tuple:
    #             - the loss
    #             - the sample size, which is used as the denominator for the
    #               gradient
    #             - logging outputs to display while training
    #     """
        model.train()
        model.set_num_updates(update_num)



        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                #loss, sample_size, logging_output = criterion(model, sample)
                task_ids = torch.unique(sample['net_input']['src_tokens'][:,0]).tolist()
                #fixed_order
                task_ids = sorted(task_ids)

                if self.dirs_keep!=None:
                    keep_task_ids = []
                    for i in task_ids:
                        if self.dicts['en_XX'][i] in self.dirs_keep:
                            keep_task_ids.append(i)
                    if len(keep_task_ids)==0:
                        task_ids = [task_ids[0]]
                    else:
                        task_ids = keep_task_ids
                
                sample_id_groups = [(sample['net_input']['src_tokens'][:,0]==i).nonzero(as_tuple=True)[0] for i in task_ids]
                sample_groups = split_samples_by_tasks(sample,sample_id_groups)
                loss_tuples = [criterion(model, i)for i in sample_groups]
        

        # Compute Gradient Properties For Each Task

        if self.log_gradient_var and update_num%100==0 and update_num not in self.logged_step:
            Gradient_Properties = logTaskGradientProperty(task_ids,sample_groups,criterion,model,optimizer)
            logger.info("Proerty Each Task, Current Step | %d | "%update_num+str(Gradient_Properties))
            self.logged_step.add(update_num)

        # Conduct Gradient Backward
        losses= [i[0] for i in loss_tuples]
        loss=sum(losses).item()
        
        sample_size=loss_tuples[0][2]['ntokens']
        nll_losses=loss_tuples[0][2]['nll_loss']

        logging_output={"loss":loss,"ntokens":sample_size,"sample_size":sample_size,"nll_loss":nll_losses}

    #     if self.conflict_drop_task!=None:
    #         assert self.conflict_drop_layers != None and self.conflict_drop_layers != [], "conflict layers must be set"
    #         task_drop_list = []
    #         for i in task_ids:
    #             if self.dicts['en_XX'][i] in self.conflict_drop_task:
    #                 task_drop_list.append(1)
    #             else:
    #                 task_drop_list.append(0)
    #         if self.use_task_drop:
    #             TaskDrop(losses,task_drop_list,self.conflict_drop_layers,optimizer,self.conflict_drop_prob)
    #         else:
    #             ConflictDrop(losses,task_drop_list,self.conflict_drop_layers,optimizer,self.conflict_drop_prob)
    #     else:
        
        with torch.autograd.profiler.record_function("backward"):
            for task_id,task_loss in zip(task_ids,losses):
                # HACK zh_cn_low_weight0.1

                # if task_id == 64027:
                #     task_loss *= 0.1

                optimizer.backward(task_loss)
        
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):

        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        # import pdb
        # pdb.set_trace()
        return loss, sample_size, logging_output




    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    src_tokens = sample["net_input"]["src_tokens"]
                    bsz = src_tokens.size(0)
                    prefix_tokens = (
                        torch.LongTensor([[tgt_lang_tok]]).expand(bsz, 1).to(src_tokens)
                    )
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    constraints=constraints,
                )
            else:
                return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=self.data_manager.get_decoder_langtok(
                        self.args.target_lang, tgt_langtok_spec
                    )
                    if tgt_langtok_spec
                    else self.target_dictionary.eos(),
                )

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.data_manager.get_source_dictionary(self.source_langs[0])

    @property
    def target_dictionary(self):
        return self.data_manager.get_target_dictionary(self.target_langs[0])

    def create_batch_sampler_func(
        self,
        max_positions,
        ignore_invalid_inputs,
        max_tokens,
        max_sentences,
        required_batch_size_multiple=1,
        seed=1,
    ):
        def construct_batch_sampler(dataset, epoch):
            splits = [
                s for s, _ in self.datasets.items() if self.datasets[s] == dataset
            ]
            split = splits[0] if len(splits) > 0 else None
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
            logger.info(
                f"[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # filter examples that are too large
            if max_positions is not None:
                my_time = time.time()
                indices = self.filter_indices_by_size(
                    indices, dataset, max_positions, ignore_invalid_inputs
                )
                logger.info(
                    f"[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}"
                )
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # create mini-batches with given size constraints
            my_time = time.time()
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            logger.info(
                f"[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}"
            )
            logger.info(
                f"[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}"
            )
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            return batch_sampler

        return construct_batch_sampler

    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]
        if self.args.sampling_method == "RoundRobin":
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter
        
        # HACK adding multiple batch sampler for different directions
        
        if isinstance(dataset.datasets,dict): # for training batches
            construct_batch_samplers = {
                i:self.create_batch_sampler_func(
                max_positions,
                ignore_invalid_inputs,
                j,
                max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed)
                for i,j in eval(self.data_manager.args.recipe).items()
            }
            # HACK adding multiple epoch sampler for different directions
            first_key = list(dataset.datasets.keys())[0]
            epoch_iter = iterators.HotpotIterator(
                dataset.datasets,
                dataset.datasets[first_key].collater,
                construct_batch_samplers,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch
            )
            

        else: # for valid batches
            construct_batch_sampler = self.create_batch_sampler_func(
                max_positions,
                ignore_invalid_inputs,
                max_tokens,
                max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
            )

            epoch_iter = iterators.EpochBatchIterator(
                dataset=dataset,
                collate_fn=dataset.collater,
                batch_sampler=construct_batch_sampler,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
            )
        


        return epoch_iter



def split_samples_by_tasks(sample,task_ids:list):
    ret = []
    number_of_tasks = len(task_ids)
    for task_group in task_ids:
        new_sample={}
        new_sample['id'] = sample['id'].index_select(0,task_group)
        new_sample['nsentences'] = sample['nsentences']
        new_sample['ntokens'] = sample['ntokens']
        new_sample['sentences_current_task'] = len(task_group)
        new_sample['target'] = sample['target'].index_select(0,task_group)
        new_sample['net_input'] = {}
        new_sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].index_select(0,task_group)
        new_sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].index_select(0,task_group)
        new_sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].index_select(0,task_group)
        ret.append(new_sample)
    return ret

from .translation_multi_simple_epoch_multitask import get_task_gradients



def split_examples_from_batch(batch_dict):
    example_list=[]
    for i in range(len(batch_dict['id'])):
        example={}
        example['id'] = batch_dict['id'][i]
        example['nsentences'] = batch_dict['nsentences']
        example['ntokens'] = batch_dict['ntokens']
        example['sentences_current_task'] = 1
        example['target'] = batch_dict['target'][i].unsqueeze(0)
        example['net_input'] = {}
        example['net_input']['src_tokens'] = batch_dict['net_input']['src_tokens'][i].unsqueeze(0)
        example['net_input']['src_lengths'] = batch_dict['net_input']['src_lengths'][i]
        example['net_input']['prev_output_tokens'] = batch_dict['net_input']['prev_output_tokens'][i].unsqueeze(0)
        example_list.append(example)
    return example_list


KEEP_LAYER_NAMES=[
    'encoder.layers.0.fc1.weight',
    'encoder.layers.3.fc1.weight',
    'encoder.layers.5.fc1.weight',
    'decoder.layers.0.fc1.weight',
    'decoder.layers.3.fc1.weight',
    'decoder.layers.5.fc1.weight',
]

def logTaskGradientProperty(task_ids,task_examples,criterion,model,optimizer):
    gradient_property={}

    assert len(task_ids) == len(task_examples)

    for task_id,examples in zip(task_ids,task_examples):
        current_task_property = {}
        current_task_gradients_per_layer = {} 

        for example in split_examples_from_batch(examples):
            loss_per_example=criterion(model, example)[0]
            loss_per_example.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and name in KEEP_LAYER_NAMES:
                    if name not in current_task_gradients_per_layer:
                        current_task_gradients_per_layer[name] = []
                    current_task_gradients_per_layer[name].append(param.grad.detach().view(-1))
            
            optimizer.zero_grad()
        
        for para_name,para_grads in current_task_gradients_per_layer.items():
            current_task_property[para_name] = {}
            data = torch.stack(current_task_gradients_per_layer[para_name])
            current_task_property[para_name]['var'] = torch.var(data,0).mean().item()
            current_task_property[para_name]['l2_norm'] = torch.norm(data.mean(dim=0),2).item()
        
        gradient_property[task_id] = current_task_property

    return gradient_property



        






def TaskDrop(objectives:list,tasks_to_drop:list,layers_to_drop_id:list,optimizer,drop_p=0.5):
    if len(objectives)==1 or sum(tasks_to_drop)==0:
        for i in objectives:
            i.backward()
        return

    # compute gradient of each task
    grads, has_grads = get_task_gradients(objectives,optimizer)
    new_grads = []
    layer_num = len(grads[0])
    task_num = len(grads)

    # compute the projection of the gradient to the sum gradient

    for index in range(layer_num):
        if not has_grads[0][index]:
            new_grads.append(grads[0][index])
            continue

        layer_grads_per_task = [grads[i][index] for i in range(task_num)]
        if index not in layers_to_drop_id:
            new_grads.append(sum(layer_grads_per_task))
        else:
            general_gradient = sum(layer_grads_per_task)
            for whether_drop, gradient in zip(tasks_to_drop, layer_grads_per_task):
                if whether_drop and torch.rand(1).item()<drop_p:
                   # compute the project vector from gradient to general gradient
                   general_gradient -= gradient
            new_grads.append(general_gradient)
    
    # update the optimizer with the processed gradients
    optimizer._set_grad(new_grads)




def ConflictDrop(objectives:list,tasks_to_drop:list,layers_to_drop_id:list,optimizer,drop_p=0.5):
    # first compute gradient of each task
    # second compute the projection of the gradient to the sum gradient
    # third drop the corresponding conflict gradients of the tasks_to_drop_id in layers_to_drop_id
    # fourth return the processed gradients
    # fifth update the optimizer with the processed gradients
    if len(objectives)==1 or sum(tasks_to_drop)==0:
        for i in objectives:
            i.backward()
        return
    
    # compute gradient of each task
    grads, has_grads = get_task_gradients(objectives,optimizer)
    new_grads = []
    layer_num = len(grads[0])
    task_num = len(grads)

    # compute the projection of the gradient to the sum gradient

    for index in range(layer_num):
        if not has_grads[0][index]:
            new_grads.append(grads[0][index])
            continue

        layer_grads_per_task = [grads[i][index] for i in range(task_num)]
        if index not in layers_to_drop_id:
            new_grads.append(sum(layer_grads_per_task))
        else:
            general_gradient = sum(layer_grads_per_task)
            for whether_drop, gradient in zip(tasks_to_drop, layer_grads_per_task):
                if whether_drop and torch.rand(1).item()<drop_p:
                   # compute the project vector from gradient to general gradient
                   project_vector = ( torch.mul(general_gradient,gradient) / (sum(general_gradient**2) + 1e-12) ) *general_gradient
                   droping_gradient = gradient - project_vector
                   general_gradient -= droping_gradient
            
            new_grads.append(general_gradient)
    
    # update the optimizer with the processed gradients
    optimizer._set_grad(new_grads)
            
