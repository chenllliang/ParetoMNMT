# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import pdb
import datetime
import logging
import time
from fairseq.optim.amp_optimizer import AMPOptimizer
import random,os,json

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
import math
from torch.nn.functional import cosine_similarity

from itertools import combinations

###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()


###


logger = logging.getLogger(__name__)





@register_task("translation_multi_simple_epoch_dpl")
class TranslationMultiSimpleEpochTaskDPL(LegacyFairseqTask):
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

    basic loop for multitask curriculum MNMT:

    tasks_weight = {i:1 for i in tasks}
    multitask_validation_loss_pre = {i:0 for i in tasks}

    for i,j in enumerate(dataloader):
        model.train_step(j,task_weight)

        if i % validation_steps==0:
            multitask_validation_loss_pre = multitask_validation_loss
            multitask_validation_loss = [model.validate_step(validation_split) for validation_split in valiation_datasets]

            # reweight the tasks weight
            task_weight = curriculum(tasks_weight, multitask_validation_loss_pre, multitask_validation_loss)

    """



    def compute_weight_base_T(self,sizes,t):

        original_weights = [i/sum(sizes) for i in sizes]
        
        temped_weights = [i**(1/t) for i in original_weights]

        task_weighted = [i/sum(temped_weights) for i in temped_weights]

        return task_weighted


    def update_weight_first_softmax(self):
        ema_alpha = self.curriculum_ema
        normlize_T = self.curriculum_T

        if self.val_steps < 2: # "first_derivative method need at least two validation steps"
            return
        else:
            delta_dev_loss = { i:j[-1]/j[-2] for i,j in self.curriculum_dev_loss.items()}  # loss scale change, bigger means slower, smaller means faster
            losses = list(delta_dev_loss.values())

            N = len(losses)

            softmax_losses = [ math.exp(i/normlize_T) for i in losses ]


            task_weighted = [i*N/sum(softmax_losses) for i in softmax_losses]

            new_weight = {}
            for i,j in enumerate(self.curriculum_weight.keys()):
                new_weight[j] = self.curriculum_weight[j]*(1-ema_alpha) + ema_alpha*task_weighted[i]
            
            self.curriculum_weight = new_weight

            return new_weight






    def update_cl_weight_first_derivative(self):
        # using delta of dev loss changefor each task to decide the weight for next loop
        # first derivatives

        # loss smoothing by 0.1, maybe too large
        ema_alpha = self.curriculum_ema
        normlize_T = self.curriculum_T

        if self.val_steps < 2: # "first_derivative method need at least two validation steps"
            return
        else:
            delta_dev_loss = { i:j[-2]-j[-1] for i,j in self.curriculum_dev_loss.items()}
            losses = list(delta_dev_loss.values())
            min_loss = min(losses)
            max_loss = max(losses)

            
            #ori norm_losses = [(i-min_loss+0.1)/(max_loss-min_loss) for i in losses]

            if min_loss > 0:
                norm_losses = losses
            else:
                # min_max_normalization
                norm_losses = [(i-min_loss+0.001)/(max_loss-min_loss) for i in losses]

            origin_weight = [i/sum(norm_losses) for i in norm_losses]
            temp_weights = [ i**(1/normlize_T)  for i in norm_losses]
            temp_norm_weights = [ i/sum(temp_weights) for i in temp_weights]


            task_weighted = [(i/sum(temp_norm_weights))*len(temp_norm_weights) for i in temp_norm_weights]

            # reweight curriculum weight

            new_weight = {}
            for i,j in enumerate(self.curriculum_weight.keys()):
                new_weight[j] = self.curriculum_weight[j]*(1-ema_alpha) + ema_alpha*task_weighted[i]
            

            self.curriculum_weight = new_weight

            return new_weight





        # softmax normalization





        





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
        parser.add_argument('--is-curriculum', action='store_true',default=False,
                            help='whether to use curriculum')

        parser.add_argument('--is-dtd', action='store_true',default=False,
                            help='whether to use dynamic task dropping')
        parser.add_argument('--max-dtd-p', default=0.3)
        parser.add_argument('--dtd-keep-norm', action='store_true',default=False,
                                    help='whether to keep task norm')
        parser.add_argument('--dtd-p',default=None)

        


        parser.add_argument('--log-task-valid-loss', action='store_true',default=False,
                            help='whether log task valid loss')

        parser.add_argument('--curriculum-ema', default=0.1)
        parser.add_argument('--curriculum-T', default=5)

        parser.add_argument('--pcgrad', default=None)
        parser.add_argument('--grad-drop', action='store_true',default=False)

        parser.add_argument('--init-weight-T', default=None)
        parser.add_argument('--init-weight-datasize', default=None)

        parser.add_argument('--second-d', action='store_true',default=False)



        parser.add_argument('--gradient-saving', action='store_true',default=False)
        parser.add_argument('--gradient-saving-path', default=None)


        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
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

        self.curriculum_ema = float(args.curriculum_ema)
        self.curriculum_T = float(args.curriculum_T)

        #Optmization Method
        self.pcgrad = float(args.pcgrad) if args.pcgrad!=None else None
        self.grad_drop = args.grad_drop 

        self.second_d = args.second_d



        # Signal for the trainer to conduct special valid step
        self.is_curriculum = args.is_curriculum
        self.log_task_valid_loss = args.log_task_valid_loss

        self.curriculum_weight = { self.dicts['en_XX'].index("__%s__"%i):1 for i in self.target_langs}
        self.curriculum_dev_loss = { self.dicts['en_XX'].index("__%s__"%i):[0] for i in self.target_langs}

        self.is_dtd = args.is_dtd
        self.max_dtd_p = float(args.max_dtd_p)
        self.dtd_keep_norm = args.dtd_keep_norm
        
        if args.dtd_p!=None:
            self.dtd_p = eval(args.dtd_p)
        else:
            self.dtd_p = { self.dicts['en_XX'].index("__%s__"%i):0 for i in self.target_langs}



        self.gradient_saving = self.args.gradient_saving
        self.gradient_saving_path = args.gradient_saving_path

        self.val_steps = 0
        self.temp_multitask_dev_loss = {}
        self.temp_multitask_dev_output_ntokens = {}


        if self.args.init_weight_T != None:
            self.init_weight_T = float(self.args.init_weight_T)
            self.datasize = eval(self.args.init_weight_datasize)

            temped_weights = self.compute_weight_base_T(list(self.datasize.values()),self.init_weight_T)
            print("weights_temperature: ",temped_weights)

            for i,j in enumerate(self.datasize.keys()):
                self.curriculum_weight[self.dicts['en_XX'].index("__%s__"%j)] = temped_weights[i]
        else:
            self.init_weight_T = None



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
                #loss, sample_size, logging_output = criterion(model, sample)

                task_ids = torch.unique(sample['net_input']['src_tokens'][:,0]).tolist()
                task_ids = sorted(task_ids)

                
                sample_id_groups = [(sample['net_input']['src_tokens'][:,0]==i).nonzero(as_tuple=True)[0] for i in task_ids]
                sample_groups = split_samples_by_tasks(sample,sample_id_groups)


                num_tokens_each_task = [i.item() for i in grep_number_of_tokens(sample_groups)]

                loss_tuples = [criterion(model, i)for i in sample_groups]


            # saving gradients

            if self.gradient_saving and update_num%100==0:
                Gradient_Properties = SaveTaskGradient(task_ids,sample_groups,criterion,model,optimizer)
                with open(os.path.join(self.gradient_saving_path,str(update_num)+".gradient"), 'w') as f:
                    f.write(json.dumps(Gradient_Properties))


            # Conduct Gradient Backward
            losses= [i[0] for i in loss_tuples]
            loss=sum(losses).item()
            
            sample_size=loss_tuples[0][2]['ntokens']
            nll_losses=loss_tuples[0][2]['nll_loss']

            logging_output={"loss":loss,"ntokens":sample_size,"sample_size":sample_size,"nll_loss":nll_losses}



            
            if self.pcgrad != None and len(losses)>=2:
                gradient_vaccine_multi_obj(losses,self.pcgrad,optimizer)
            
            elif self.grad_drop != False and len(losses)>=2:
                grad_drop(losses,optimizer)

            elif self.init_weight_T != None and len(losses)>=2:

                online_batch_weight = [i/sum(num_tokens_each_task) for i in num_tokens_each_task]
                temped_weight = self.compute_weight_base_T(online_batch_weight,self.init_weight_T)

                total_loss = []

                with torch.autograd.profiler.record_function("backward"):
                    for index,task_loss in enumerate(losses):
                        weighted_task_loss = (task_loss*temped_weight[index])/online_batch_weight[index]
                        # keep gradient norm the same
                        total_loss.append(weighted_task_loss)
                    
                    original_sum = loss
                    new_sum = sum(total_loss).item()
                    normlized_weighted_task_loss = sum(total_loss)*(original_sum/new_sum)
                    
                    optimizer.backward(normlized_weighted_task_loss)

            else:
                with torch.autograd.profiler.record_function("backward"):
                    #hack dtd
                    for task_id,task_loss in zip(task_ids,losses):
                        weighted_task_loss = task_loss*self.curriculum_weight[task_id]
                        optimizer.backward(weighted_task_loss)
    
            
            return loss, sample_size, logging_output

    def per_task_valid_step(self,sample,model,criterion):
        task_ids = torch.unique(sample['net_input']['src_tokens'][:,0]).tolist()
        task_ids = sorted(task_ids)
        sample_id_groups = [(sample['net_input']['src_tokens'][:,0]==i).nonzero(as_tuple=True)[0] for i in task_ids]
        sample_groups = split_samples_by_tasks(sample,sample_id_groups)

        tokens_per_target = grep_number_of_tokens(sample_groups)
        loss_tuples = [self.valid_step(i, model, criterion) for i in sample_groups]

        assert len(task_ids)==len(tokens_per_target)==len(loss_tuples)

        for i,j in enumerate(task_ids):
            if j in self.temp_multitask_dev_loss:
                self.temp_multitask_dev_loss[j]+=loss_tuples[i][2]['nll_loss'].item()
                self.temp_multitask_dev_output_ntokens[j]+=tokens_per_target[i].item()
            else:
                self.temp_multitask_dev_loss[j]=loss_tuples[i][2]['nll_loss'].item()
                self.temp_multitask_dev_output_ntokens[j]=tokens_per_target[i].item()
            
    def per_task_valid_step_sharpness(self,sample,model,criterion):
        task_ids = torch.unique(sample['net_input']['src_tokens'][:,0]).tolist()
        task_ids = sorted(task_ids)
        sample_id_groups = [(sample['net_input']['src_tokens'][:,0]==i).nonzero(as_tuple=True)[0] for i in task_ids]
        sample_groups = split_samples_by_tasks(sample,sample_id_groups)

        tokens_per_target = grep_number_of_tokens(sample_groups)
        loss_tuples = [self.valid_step(i, model, criterion) for i in sample_groups]

        if self.second_d:
            second_ds = [self.compute_sharpness_one_task(i,model,criterion) for i in sample_groups]
            print(second_ds)

    def compute_sharpness_one_task(self,sample,model,criterion):
        loss = criterion(model, sample)[0]

        loss.backward(retain_graph=True, create_graph=True)
        loss.backward()

        param_sharpness = {}

        for name, param in model.named_parameters():
            if param.grad is not None and name in KEEP_LAYER_NAMES:
                param_sharpness[name]=torch.sum(param.grad.long().abs()).detach().view(-1).cpu().numpy().tolist()
        
        
        
        return param_sharpness

        


        

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

def grep_number_of_tokens(task_groups):
    number_of_target_tokens = []
    for i in task_groups:
        number_of_target_tokens.append(torch.sum(i['target']!=1))
    return number_of_target_tokens


def gradient_vaccine_multi_obj(objectives,alpha,optimizer,return_cosine=False):
    # make cosine similarity between grad1 and grad2 to alpha
    # alpha is between 0 and 1
    alpha = torch.tensor(alpha).cuda()

    assert len(objectives)  >= 2

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

    # generate a deepcopy of grads
    old_grads = copy.deepcopy(grads)

    grads_ids = list(range(len(grads)))

    grads_combinations = list(combinations(grads_ids,2))

    for (task1,task2) in grads_combinations:
        for index,(grad1,grad2) in enumerate(zip(grads[task1],grads[task2])):
            # g1' = g1 + a2*g2
            # g_new = g1' + g2
            if not has_grads[task1][index] and not has_grads[task2][index]:
                continue
            else:
                original_shape = grad1.shape
                grad1 = grad1.view(-1)
                grad2 = grad2.view(-1)

                cos_sim = cosine_similarity(grad1,grad2,dim=0)
                if cos_sim > alpha:
                    grad1 = grad1 + alpha*grad2
                else:
                    grad1 = grad1 + (1-cos_sim)*grad2
                grad1 = grad1.view(original_shape)
                # set the new gradient of task1
                old_grads[task1][index] = grad1
    
    # sum the gradient in old_grads with same index
    for index in range(len(old_grads[0])):
        grad_sum = 0
        for task in range(len(old_grads)):
            grad_sum += old_grads[task][index]
        new_grads.append(grad_sum)


    optimizer._set_grad(new_grads)
    return 

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

            else:
                new_grads.append((grad1+grad2).view(original_shape))


    # set new gradients
    optimizer._set_grad(new_grads)
    return 

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

def grad_drop(objectives:list,optimizer):
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

KEEP_LAYER_NAMES=[
    'encoder.layers.0.fc1.weight',
    'encoder.layers.2.fc1.weight',
    'decoder.layers.0.fc1.weight',
    'decoder.layers.2.fc1.weight',
]

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

def SaveTaskGradient(task_ids,task_examples,criterion,model,optimizer):
    gradient_property={}
    assert len(task_ids) == len(task_examples)
    for task_id,examples in zip(task_ids,task_examples):
        current_task_gradients_per_layer = {} 
        loss_per_task = criterion(model, examples)[0]
        loss_per_task.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and name in KEEP_LAYER_NAMES:
                current_task_gradients_per_layer[name]=param.grad.detach().view(-1).cpu().numpy().tolist()
        optimizer.zero_grad()
        gradient_property[task_id] = current_task_gradients_per_layer

    return gradient_property

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
