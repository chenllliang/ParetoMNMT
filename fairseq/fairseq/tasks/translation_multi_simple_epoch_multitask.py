# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import datetime
import logging
import time
from fairseq.optim.amp_optimizer import AMPOptimizer
import torch
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    iterators,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.utils import FileContentsAction
import torch.nn as nn
import json,random
from fairseq.logging import metrics
from torch.nn.functional import cosine_similarity
from .pcgrad import PCGrad


random.seed(222)

###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


###


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch_multitask")
class TranslationMultiSimpleEpochMultiTask(LegacyFairseqTask):
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
        
        parser.add_argument('--pcgrad',default=False, action='store_true',
                            help='use PCGrad during loss backward, task divided by target language')

        parser.add_argument('--computegrad',default=False, action='store_true',
                            help='compute multitask grad confilict')

        parser.add_argument('--grad-vaccine',default=False, action='store_true',
                            help='whether to use grad vaccine')
        
        parser.add_argument('--grad-vaccine-alpha',default=0.5, type=float,
                            help='alpha for grad vaccine')

        parser.add_argument('--grad-vaccine-ema-beta',default=0, type=float,
                            help='beta for ema of grad vaccine')
        
        parser.add_argument('--grad-drop',default=False, action='store_true',
                            help='whether to use grad drop')
        
        parser.add_argument('--grad-dir',default=None,
                            help='dir to store grad_cos files ')

        parser.add_argument('--loss-temperature',default=0,
                            help='loss balancing temperature ')

        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
        self.pcgrad = args.pcgrad
        self.computegrad = args.computegrad
        self.grad_dir = args.grad_dir
        self.gradient_vaccine = args.grad_vaccine
        self.grad_vaccine_alpha = args.grad_vaccine_alpha
        self.ema_beta = args.grad_vaccine_ema_beta
        self.grad_drop = args.grad_drop
        self.loss_temperature = args.loss_temperature

        if self.ema_beta!=0:
            self.old_alphas = self.grad_vaccine_alpha
        
        self.log_freq=100
        self.langs = langs
        self.dicts = dicts
        self.training = training
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
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)



    
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):

                if self.pcgrad:
                    #enable pcgrad, divide samples by target id (first input token)
                    task_ids = torch.unique(sample['net_input']['src_tokens'][:,0]).tolist()
                    sample_id_groups = [(sample['net_input']['src_tokens'][:,0]==i).nonzero(as_tuple=True)[0] for i in task_ids]
                    sample_groups = split_samples_by_tasks(sample,sample_id_groups)
                    loss_tuples = [criterion(model, i)for i in sample_groups]              
                
                else:
                    #loss, sample_size, logging_output = criterion(model, sample)
                    task_ids = torch.unique(sample['net_input']['src_tokens'][:,0]).tolist()
                    #fixed_order
                    task_ids = sorted(task_ids)
                    sample_id_groups = [(sample['net_input']['src_tokens'][:,0]==i).nonzero(as_tuple=True)[0] for i in task_ids]
                    sample_groups = split_samples_by_tasks(sample,sample_id_groups)
                    loss_tuples = [criterion(model, i)for i in sample_groups]   

                    #loss, sample_size, logging_output

                    #HACK

                    if update_num%self.args.log_interval==0:
                        logger.info("sample_size:"+str([len(i) for i in sample_id_groups]))
                        logger.info("loss:"+str([i[0].item() for i in loss_tuples]))
                        logger.info("loss/sample_size:"+str([i[0].item()/len(sample_id_groups[j]) for j,i in enumerate(loss_tuples)]))
                    

        
        if self.computegrad and update_num%self.args.log_interval==0:
            init_param_grad = {}
            for name,param in model.named_parameters():
                if "weight" in name:
                    init_param_grad[name] = torch.zeros_like(param)
            task_gradients={}
        
        if self.pcgrad:
            #merge with gradient vaccine
            pass
        
        elif self.loss_temperature!=0:
            losses= [i[0] for i in loss_tuples]
            loss=sum(losses).item()
            sample_size=loss_tuples[0][2]['ntokens']
            nll_losses=loss_tuples[0][2]['nll_loss']
            logging_output={"loss":loss,"ntokens":sample_size,"sample_size":sample_size,"nll_loss":nll_losses}
            
            loss_balance(losses,list(range(0,143)),5,optimizer,None)

        elif self.grad_drop:
            losses= [i[0] for i in loss_tuples]
            loss=sum(losses).item()
            sample_size=loss_tuples[0][2]['ntokens']
            nll_losses=loss_tuples[0][2]['nll_loss']
            logging_output={"loss":loss,"ntokens":sample_size,"sample_size":sample_size,"nll_loss":nll_losses}
            grad_drop(losses,list(range(1,143)),optimizer)
             
            
        elif self.gradient_vaccine and len(loss_tuples)>=2:
            losses= [i[0] for i in loss_tuples]
            loss=sum(losses).item()
            
            sample_size=sum([i[2]['ntokens']for i in loss_tuples])
            nll_losses=sum([i[2]['nll_loss']for i in loss_tuples])

            logging_output={"loss":loss,"ntokens":sample_size,"sample_size":sample_size,"nll_loss":nll_losses}
            
            if self.ema_beta!=0:
                current_alpha, task_pair_cosine = gradient_vaccine_2_obj_with_ema(losses,self.old_alphas,optimizer,self.ema_beta)
                self.old_alphas = current_alpha
                if update_num%self.args.log_interval==0: 
                    with open(self.grad_dir+"/task_pair_cosine_%d.json"%(update_num), "w") as f:
                        json.dump(task_pair_cosine, f)
                    with open(self.grad_dir+"/task_pair_ema_alphas_%d.json"%(update_num), "w") as f:
                        json.dump(current_alpha, f)

            
            else:
                if update_num%self.args.log_interval==0: 
                    task_pair_cosine = gradient_vaccine_2_obj(losses,self.grad_vaccine_alpha,optimizer,return_cosine=True)
                    with open(self.grad_dir+"/task_pair_cosine_%d.json"%(update_num), "w") as f:
                        json.dump(task_pair_cosine, f)
                else:
                    task_pair_cosine = gradient_vaccine_2_obj(losses,self.grad_vaccine_alpha,optimizer,return_cosine=False)




            
        
        else:

            losses= [i[0] for i in loss_tuples]
            loss=sum(losses).item()
            
            sample_size=loss_tuples[0][2]['ntokens']
            nll_losses=loss_tuples[0][2]['nll_loss']
            #n_sentences=sum([i[2]['sentences_current_task']for i in loss_tuples])

            logging_output={"loss":loss,"ntokens":sample_size,"sample_size":sample_size,"nll_loss":nll_losses}
            

            
            # torch.cuda.synchronize()
            # start_backward = time.time()

            with torch.autograd.profiler.record_function("backward"):
                for task_id,task_loss in zip(task_ids,losses):
                    optimizer.backward(task_loss)
                    #x.grad += dloss/dx

                    if self.computegrad and update_num%self.args.log_interval==0:
                        #compute gradent for each task for each layer
                        current_named_gradient={}
                        for name,param in model.named_parameters():
                            if name in init_param_grad.keys():
                                if param.grad == None:
                                    continue
                                else:
                                    current_named_gradient[name] = param.grad.detach() - init_param_grad[name]
                                    init_param_grad[name] = param.grad.detach() - torch.zeros_like(param)
                        task_gradients[task_id] = current_named_gradient
            

            if self.computegrad and update_num%self.args.log_interval==0:
                # compute layer-wise and taskpair wise cosine similarity

                task_pair_cosine={}

                for i in range(len(task_gradients)-1):
                    for j in range(i+1,len(task_gradients)):
                        key_name=list(self.dicts.keys())[0]
                        task_pair_name="%s-%s"%(self.dicts[key_name][task_ids[i]],self.dicts[key_name][task_ids[j]])
                        task_pair_cosine[task_pair_name]={}

                        for name,param in task_gradients[task_ids[i]].items():
                            if name in task_gradients[task_ids[j]].keys():
                                task_pair_cosine[task_pair_name][name]=round(cosine_similarity(param.view(-1),task_gradients[task_ids[j]][name].view(-1),dim=0).item(),4)

                        
                        with open(self.grad_dir+"/task_pair_cosine_%d.json"%(update_num), "w") as f:
                            json.dump(task_pair_cosine, f)
                


    
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
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

def get_task_gradients(objectives,optimizer):
    grads, has_grads = [], []
    for index,obj in enumerate(objectives):
        optimizer.zero_grad()
        
        if index==len(objectives)-1:
            obj.backward()
        else:
            obj.backward(retain_graph=True)
        
        grad, has_grad = optimizer._retrieve_grad()
        grads.append(grad)
        has_grads.append(has_grad)
    return grads, has_grads



def self_pace_balance(objectives,deltas,optimizer):
    pass


def loss_balance(objectives,layers_to_balance,temperature,optimizer,sample_id_groups=None):
    # compute down/up sampling rate for different loss
    # wrong method, we should look at dev loss to decide how to balance

    losses = torch.stack(objectives)


    if sample_id_groups:
        number_of_examples_per_task = torch.tensor([len(i) for i in sample_id_groups]).cuda()
        per_instance_losses=losses/number_of_examples_per_task
        loss_sum = sum(per_instance_losses).detach()
        original_ratio = per_instance_losses.detach()/loss_sum

    else:
        loss_sum = sum(losses).detach()
        original_ratio = losses.detach()/loss_sum

    new_portion = original_ratio**(1/temperature)
    new_ratio = new_portion / sum(new_portion)
    new_loss = (losses*new_ratio) / original_ratio
    optimizer.backward(sum(new_loss))


def grad_drop(objectives:list,layers_to_drop_id:list,optimizer):
    # 1. compute Gradient Positive Sign Purity
    # 2. generate gradient mask
    # 3. apply mask
    # 4. renew gradient, normalize, and apply to optimizer

    if len(objectives)==1:
        objectives[0].backward()
        return
    grads, has_grads = get_task_gradients(objectives,optimizer)
    new_grads = []
    layer_num = len(grads[0])
    task_num = len(grads)

    for index in range(layer_num):
        if not has_grads[0][index]:
            new_grads.append(grads[0][index])
            continue

        layer_grads_per_task = [grads[i][index] for i in range(task_num)]
        if index not in layers_to_drop_id:
            new_grads.append(sum(layer_grads_per_task))
        else:
            # 1. Compute Gradient Positive Sign Purity
            
            grad_abs_sum = sum([torch.abs(grad) for grad in layer_grads_per_task]) # same as gradient size
            pos_sign_purity = 0.5 + sum(layer_grads_per_task) / (2. * grad_abs_sum + 1e-7) # same as gradient size

            pos_grad = sum([grad*(grad>=0) for grad in layer_grads_per_task]) # same as gradient size
            neg_grad = sum([grad*(grad<0) for grad in layer_grads_per_task]) # same as gradient size
            
            U = torch.rand_like(pos_sign_purity)
            pos_mask = pos_sign_purity >= U
            neg_mask = pos_sign_purity < U
            new_grads.append(pos_mask*pos_grad + neg_mask*neg_grad)
    
    optimizer._set_grad(new_grads)


def gradient_vaccine_2_obj_with_ema(objectives,alpha,optimizer,beta=0.01,return_cosine=True):
    # make cosine similarity between grad1 and grad2 to alpha dict
    # each layer has different alpha
    # alpha is between 0 and 1
    assert len(objectives)  == 2

    # get the gradient of the whole model from different objectives
    grads, has_grads = [], []
    for index,obj in enumerate(objectives):
        optimizer.zero_grad()
        
        if index==len(objectives)-1:
            obj.backward()
        else:
            obj.backward(retain_graph=True)
        
        grad, has_grad = optimizer._retrieve_grad()
        grads.append(grad)
        has_grads.append(has_grad)
    
    # compute new gradients only for 2 tasks
    new_grads=[]
    random.shuffle(grads)

    # make init alpha a dict
    if isinstance(alpha,float):
        new_alpha={}
        for index , _ in enumerate(grads[0]):
            new_alpha[index] = alpha
    elif isinstance(alpha,dict):
        new_alpha = alpha
    else:
        raise ValueError("alpha should be float or dict")

    alphas_current={}
    cos_original={}
    for index,(grad1,grad2) in enumerate(zip(grads[0],grads[1])):
        # g1' = g1 + a2*g2
        # g_new = g1' + g2
        if not has_grads[0][index] and not has_grads[1][index]:
            new_grads.append(grad1)
            cos_original[index] = 0
            continue

        elif not has_grads[0][index] and has_grads[1][index]:
            raise RuntimeError("No gradient for task 1 but gradient for task 2")
        elif has_grads[0][index] and not has_grads[1][index]:
            raise RuntimeError("No gradient for task 2 but gradient for task 1")
        
        else:
            original_shape = grad1.shape
            grad1 = grad1.view(-1)
            grad2 = grad2.view(-1)

            cos_sim = cosine_similarity(grad1,grad2,dim=0)
            cos_original[index]=cos_sim.item()

            # ema
            current_alpha = new_alpha[index]*(1-beta) + beta*cos_sim
            alphas_current[index] = current_alpha.item()

            if cos_sim < current_alpha:
                # compute a2 for g2
                a2 = grad1.norm(2) * (current_alpha*torch.sqrt(1-cos_sim**2) - cos_sim*torch.sqrt(1-current_alpha**2))
                a2 /= (grad2.norm(2) * torch.sqrt(1-current_alpha**2))

                g1_new = grad1 + a2*grad2
                g_new = g1_new + grad2

                new_grads.append(g_new.view(original_shape))
            else:
                new_grads.append((grad1+grad2).view(original_shape))


    # set new gradients
    optimizer._set_grad(new_grads)
    return alphas_current,cos_original

def gradinet_vaccine_multi_obj_ema(objectives:dict,alpha,optimizer,beta=0.01,return_cosine=True):
    # get the gradient of the whole model from different objectives
    grads, has_grads = [], []
    for index,obj in enumerate(objectives.values()):
        optimizer.zero_grad()
        
        if index==len(objectives)-1:
            obj.backward()
        else:
            obj.backward(retain_graph=True)
        
        grad, has_grad = optimizer._retrieve_grad()
        grads.append(grad)
        has_grads.append(has_grad)
    
    #HACK
    
    # compute edited gradients for all tasks
    new_grads=[]
    random.shuffle(grads)

    # make init alpha a dict
    if isinstance(alpha,float):
        new_alpha={}
        for index , _ in enumerate(grads[0]):
            new_alpha[index] = alpha
    elif isinstance(alpha,dict):
        new_alpha = alpha
    else:
        raise ValueError("alpha should be float or dict")

    alphas_current={}
    cos_original={}
    for index,(grad1,grad2) in enumerate(zip(grads[0],grads[1])):
        # g1' = g1 + a2*g2
        # g_new = g1' + g2
        if not has_grads[0][index] and not has_grads[1][index]:
            new_grads.append(grad1)
            cos_original[index] = 0
            continue

        elif not has_grads[0][index] and has_grads[1][index]:
            raise RuntimeError("No gradient for task 1 but gradient for task 2")
        elif has_grads[0][index] and not has_grads[1][index]:
            raise RuntimeError("No gradient for task 2 but gradient for task 1")
        
        else:
            original_shape = grad1.shape
            grad1 = grad1.view(-1)
            grad2 = grad2.view(-1)

            cos_sim = cosine_similarity(grad1,grad2,dim=0)
            cos_original[index]=cos_sim.item()

            # ema
            current_alpha = new_alpha[index]*(1-beta) + beta*cos_sim
            alphas_current[index] = current_alpha.item()

            if cos_sim < current_alpha:
                # compute a2 for g2
                a2 = grad1.norm(2) * (current_alpha*torch.sqrt(1-cos_sim**2) - cos_sim*torch.sqrt(1-current_alpha**2))
                a2 /= (grad2.norm(2) * torch.sqrt(1-current_alpha**2))

                g1_new = grad1 + a2*grad2
                g_new = g1_new + grad2

                new_grads.append(g_new.view(original_shape))
            else:
                new_grads.append((grad1+grad2).view(original_shape))


    # set new gradients
    optimizer._set_grad(new_grads)
    return alphas_current,cos_original

def gradient_vaccine_2_obj(objectives,alpha,optimizer,return_cosine=False):
    # make cosine similarity between grad1 and grad2 to alpha
    # alpha is between 0 and 1
    alpha = torch.tensor(alpha).cuda()

    assert len(objectives)  == 2

    # get the gradient of the whole model from different objectives
    grads, has_grads = [], []
    for index,obj in enumerate(objectives):
        optimizer.zero_grad()
        
        if index==len(objectives)-1:
            obj.backward()
        else:
            obj.backward(retain_graph=True)
        
        grad, has_grad = optimizer._retrieve_grad()
        grads.append(grad)
        has_grads.append(has_grad)
    
    # compute new gradients only for 2 tasks
    new_grads=[]
    random.shuffle(grads)
    cos_s=[]

    for index,(grad1,grad2) in enumerate(zip(grads[0],grads[1])):
        # g1' = g1 + a2*g2
        # g_new = g1' + g2
        if not has_grads[0][index] and not has_grads[1][index]:
            new_grads.append(grad1)
            continue

        elif not has_grads[0][index] and has_grads[1][index]:
            raise RuntimeError("No gradient for task 1 but gradient for task 2")
        elif has_grads[0][index] and not has_grads[1][index]:
            raise RuntimeError("No gradient for task 2 but gradient for task 1")
        
        else:
            original_shape = grad1.shape
            grad1 = grad1.view(-1)
            grad2 = grad2.view(-1)

            cos_sim = cosine_similarity(grad1,grad2,dim=0)


            if cos_sim < alpha:
                # compute a2 for g2
                a2 = grad1.norm(2) * (alpha*torch.sqrt(1-cos_sim**2) - cos_sim*torch.sqrt(1-alpha**2))
                a2 /= (grad2.norm(2) * torch.sqrt(1-alpha**2))

                g1_new = grad1 + a2*grad2
                g_new = g1_new + grad2

                new_grads.append(g_new.view(original_shape))

                if return_cosine:
                    cos_s.append((index,cosine_similarity(g1_new,grad2,dim=0).item()))
            else:
                new_grads.append((grad1+grad2).view(original_shape))
                if return_cosine:
                    cos_s.append((index,cos_sim.item()))

    # set new gradients
    optimizer._set_grad(new_grads)
    return cos_s



