from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from HCAN import *
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import apex 
import six 

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from tokenization_xlnet import XLNetTokenizer
from modeling_xlnet import XLNetConfig, XLNetForSequenceClassification
from optimization import AdamW, WarmupLinearSchedule
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import nltk 

import json 


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		### text_a: P, text_b: Q, text_c: O 
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.text_c = text_c
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, p_input_ids, p_input_mask, p_segment_ids, q_input_ids, q_input_mask, q_segment_ids, o_input_ids, o_input_mask, o_segment_ids, label_id):
		self.p_input_ids = p_input_ids
		self.p_input_mask = p_input_mask
		self.p_segment_ids = p_segment_ids
		self.q_input_ids = q_input_ids
		self.q_input_mask = q_input_mask
		self.q_segment_ids = q_segment_ids
		self.o_input_ids = o_input_ids
		self.o_input_mask = o_input_mask
		self.o_segment_ids = o_segment_ids
		self.label_id = label_id


class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

# convert to unicode just in case the original data is not 
def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text.decode("utf-8", "ignore")
		elif isinstance(text, unicode):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")



## TODO: RACE processor 
class dreamProcessor(DataProcessor):
	def __init__(self, data_dir):
		random.seed(42)
		self.D = [[], [], []]

		for sid in range(3):
			## Note: assuming data folder stored in the same directory 
			with open([data_dir+"/train.json", data_dir+"/dev.json", data_dir+"/test.json"][sid], "r") as f:
				data = json.load(f)
				if sid == 0:
					random.shuffle(data)
				for i in range(len(data)):
					for j in range(len(data[i][1])):
						# shouldn't do lower case, since we are using cased model 						
						# d = ['\n'.join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
						### TO TRY: instead of use nltk to sent_tokenize the passage
						### treat each turn of the dialogue as one sent 
						d = ['\n'.join(data[i][0]), data[i][1][j]["question"]]
						for k in range(len(data[i][1][j]["choice"])):
							# d += [data[i][1][j]["choice"][k].lower()]
							d += [data[i][1][j]["choice"][k]]
						# d += [data[i][1][j]["answer"].lower()] 
						d += [data[i][1][j]["answer"]] 
						self.D[sid] += [d]
		
	def get_train_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[0], "train")

	def get_test_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[2], "test")

	def get_dev_examples(self, data_dir):
		"""See base class."""
		return self._create_examples(
				self.D[1], "dev")

	def get_labels(self):
		"""See base class."""
		return ["0", "1", "2"]

	def _create_examples(self, data, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, d) in enumerate(data):
			for k in range(3):
				# each d: passage, question, choice * 3, answer 
				if data[i][2+k] == data[i][5]:
					answer = str(k)
					
			label = convert_to_unicode(answer)

			for k in range(3):
				guid = "%s-%s-%s" % (set_type, i, k)
				## passage 
				text_a = convert_to_unicode(data[i][0])
				## choice 
				text_b = convert_to_unicode(data[i][k+2])
				## question 
				text_c = convert_to_unicode(data[i][1])
				examples.append(
						InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, text_c=text_c))
			
		return examples


'''
Feature format:
p/q/o input: (input_ids, token_type_ids, attention_mask)
P shape: (no_sent, sent_len)
Q shape: (sent_len)
Opt shape: (sent_len)
'''
def convert_examples_to_features(examples, label_list, max_sent_length, max_no_sent, tokenizer, n_opt=3):
	# n_opt: number of options per question
	"""Loads a data file into a list of `InputBatch`s."""
	# Note: XLNet: A + [SEP] + B + [SEP] + C + [SEP] [CLS]
	# CLS token id for XLNet of 2 
	# pad on left for XLNet 
	cls_tok_id = 2
	pad_tok_id = 4 
	cls_token = tokenizer.cls_token
	sep_token = tokenizer.sep_token
	total_sent_len = 0 
	total_sent_no = 0
	total_question = 0  

	print("#examples:", len(examples))

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	features = [[]]
	for (ex_index, example) in enumerate(examples):
		# break passage into sentences 
		text_a = example.text_a
		sent_text_a = nltk.sent_tokenize(text_a)
		total_question += 1
		total_sent_no += len(sent_text_a)
		tokens_p = []
		for sent in sent_text_a:
			sent_tok = tokenizer.tokenize(sent)
			total_sent_len += len(sent_tok)
			tokens_p.append(sent_tok)
		# tokens_a = tokenizer.tokenize(example.text_a)

		tokens_q = None

		tokens_o = None
		
		# choice 
		if example.text_b:
			tokens_o = tokenizer.tokenize(example.text_b)

		# question 
		if example.text_c:
			tokens_q = tokenizer.tokenize(example.text_c)

		# if tokens_c:
		# 	_truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
		# 	tokens_b = tokens_c + [sep_token] + tokens_b
		# elif tokens_b:
		# 	_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		# else:
		# 	if len(tokens_a) > max_seq_length - 2:
		# 		tokens_a = tokens_a[0:(max_seq_length - 2)]

		## truncate
		if len(tokens_q) > max_sent_length - 2:
			tokens_q = tokens_q[0:(max_sent_length - 2)]
		if len(tokens_o) > max_sent_length - 2:
			tokens_o = tokens_o[0:(max_sent_length - 2)]
		for i in range(len(tokens_p)):
			if len(tokens_p[i]) > max_sent_length - 2:
				tokens_p[i] = tokens_p[i][0:(max_sent_length - 2)]

		if len(tokens_p) > max_no_sent:
			tokens_p = tokens_p[:max_no_sent]
		
		# tokens = []
		# segment_ids = []
		# for token in tokens_a:
		# 	tokens.append(token)
		# 	segment_ids.append(0)
		# tokens.append(sep_token)
		# segment_ids.append(0)

		# if tokens_b:
		# 	for token in tokens_b:
		# 		tokens.append(token)
		# 		segment_ids.append(1)
		# 	tokens.append(sep_token)
		# 	segment_ids.append(1)

		## add [SEP] [CLS] token at the end
		## make segment_ids and attention_mask for P, Q, O 
		p_input_ids = []
		p_input_mask = []
		p_segment_ids = []
		q_input_ids = []
		q_input_mask = []
		q_segment_ids = []
		o_input_ids = [] 
		o_input_mask = []
		o_segment_ids = [] 
		for i in range(len(tokens_p)):
			sent_segment_ids = [0]*len(tokens_p[i])
			tokens_p[i].append(sep_token)
			sent_segment_ids.append(0)
			tokens_p[i].append(cls_token)
			sent_segment_ids.append(cls_tok_id)
			sent_input_mask = [1]*len(tokens_p[i])
			p_segment_ids.append(sent_segment_ids)
			p_input_mask.append(sent_input_mask)
			p_input_ids.append(tokenizer.convert_tokens_to_ids(tokens_p[i]))
		q_segment_ids = [0]*len(tokens_q)
		tokens_q.append(sep_token)
		q_segment_ids.append(0)
		tokens_q.append(cls_token)
		q_segment_ids.append(cls_tok_id)
		q_input_mask = [1]*len(tokens_q)
		q_input_ids = tokenizer.convert_tokens_to_ids(tokens_q)

		o_segment_ids = [0]*len(tokens_o)
		tokens_o.append(sep_token)
		o_segment_ids.append(0)
		tokens_o.append(cls_token)
		o_segment_ids.append(cls_tok_id)
		o_input_mask = [1]*len(tokens_o)
		o_input_ids = tokenizer.convert_tokens_to_ids(tokens_o)


		# tokens.append(cls_token)
		# segment_ids.append(cls_tok_id)

		# input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# # The mask has 1 for real tokens and 0 for padding tokens. Only real
		# # tokens are attended to.
		# input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length. 
		# pad on left !!! 
		# pad_token = 0
		# pad_segment_id = 4
		
		# padding_length = max_seq_length - len(input_ids)
		# input_ids = ([0]*padding_length) + input_ids
		# input_mask = ([0]*padding_length) + input_mask
		# segment_ids = ([pad_tok_id]*padding_length) + segment_ids

		## Padding 
		for i in range(len(tokens_p)):
			padding_length = max_sent_length - len(tokens_p[i])
			p_input_ids[i] = ([0]*padding_length) + p_input_ids[i]
			p_input_mask[i] = ([0]*padding_length) + p_input_mask[i]
			p_segment_ids[i] = ([pad_tok_id]*padding_length) + p_segment_ids[i]
		## pad empty sentences to reach max_no_sent
		## pad in front 
		## pad sentences are just sentences with all padding tokens 
		if len(tokens_p) < max_no_sent:
			padding_length = max_no_sent - len(tokens_p)
			pad_sent = [0]*max_sent_length
			pad_segment_ids = [pad_tok_id]*max_sent_length
			p_input_ids = [pad_sent]*padding_length + p_input_ids
			# if len(p_input_ids) != max_sent_length:
			# 	print (p_input_ids)
			p_input_mask = [pad_sent]*padding_length + p_input_mask
			p_segment_ids = [pad_segment_ids]*padding_length + p_segment_ids 

		padding_length = max_sent_length - len(tokens_q)
		q_input_ids = ([0]*padding_length) + q_input_ids
		q_input_mask = ([0]*padding_length) + q_input_mask
		q_segment_ids = ([pad_tok_id]*padding_length) + q_segment_ids

		padding_length = max_sent_length - len(tokens_o)
		o_input_ids = ([0]*padding_length) + o_input_ids
		o_input_mask = ([0]*padding_length) + o_input_mask
		o_segment_ids = ([pad_tok_id]*padding_length) + o_segment_ids


		assert len(p_input_ids) == max_no_sent 
		assert len(p_input_mask) == max_no_sent
		assert len(p_segment_ids) == max_no_sent
		for sent in p_input_ids:
			assert len(sent) == max_sent_length, 'Wrong len: '+str(len(sent))
		for sent in p_input_mask:
			assert len(sent) == max_sent_length
		for sent in p_segment_ids:
			assert len(sent) == max_sent_length
		assert len(q_input_ids) == max_sent_length
		assert len(q_input_mask) == max_sent_length
		assert len(q_segment_ids) == max_sent_length
		assert len(o_input_ids) == max_sent_length
		assert len(o_input_mask) == max_sent_length
		assert len(o_segment_ids) == max_sent_length
		
		## Print some examples 
		label_id = label_map[example.label]
		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("passage tokens: %s" % " ".join(
					[str(x) for sent in tokens_p for x in sent]))
			logger.info("passage input_ids: %s" % " ".join([str(x) for sent in p_input_ids for x in sent]))
			logger.info("passage input_mask: %s" % " ".join([str(x) for sent in p_input_mask for x in sent]))
			logger.info(
					"passage segment_ids: %s" % " ".join([str(x) for sent in p_segment_ids for x in sent]))
			
			logger.info("question tokens: %s" % " ".join(
					[str(x) for x in tokens_q]))
			logger.info("question input_ids: %s" % " ".join([str(x) for x in q_input_ids]))
			logger.info("question input_mask: %s" % " ".join([str(x) for x in q_input_mask]))
			logger.info(
					"question segment_ids: %s" % " ".join([str(x) for x in q_segment_ids]))
			
			logger.info("option tokens: %s" % " ".join(
					[str(x) for x in tokens_o]))
			logger.info("option input_ids: %s" % " ".join([str(x) for x in o_input_ids]))
			logger.info("option input_mask: %s" % " ".join([str(x) for x in o_input_mask]))
			logger.info(
					"option segment_ids: %s" % " ".join([str(x) for x in o_segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features[-1].append(
				InputFeatures(
						p_input_ids=p_input_ids,
						p_input_mask=p_input_mask,
						p_segment_ids=p_segment_ids,
						q_input_ids=q_input_ids,
						q_input_mask=q_input_mask,
						q_segment_ids=q_segment_ids,
						o_input_ids=o_input_ids,
						o_input_mask=o_input_mask,
						o_segment_ids=o_segment_ids,
						label_id=label_id))

		## n_class egs per list 
		if len(features[-1]) == n_opt:
			features.append([])

	if len(features[-1]) == 0:
		features = features[:-1]
	print('#features', len(features))
	print ('avg no of sents per passage:', total_sent_no/total_question)
	print ('avg sent length:', total_sent_len/total_sent_no)
	return features


def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs==labels)



def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	
	parser.add_argument("--xlnet_model", default=None, type=str, required=True,
						help="Either one of the two: 'xlnet-large-cased', 'xlnet-base-cased'.")

	parser.add_argument("--output_dir",
						default=None,
						type=str,
						required=True,
						help="The output directory where the model checkpoints will be written.")


	## Other parameters
	parser.add_argument("--task",
						default="dream",
						type=str,
						help="The dataset used. Either race or dream.")
	parser.add_argument("--max_no_sent",
						default=20,
						type=int,
						help="The max number of sentences.")
	parser.add_argument("--max_sent_len",
						default=20,
						type=int,
						help="The max number of tokens in a sentence.")
	parser.add_argument("--d_model",
						default=1024,
						type=int,
						help="The hidden size of XLnet.")
	parser.add_argument("--d_lstm",
						default=1024,
						type=int,
						help="The hidden size of LSTM.")
	parser.add_argument("--lstm_layers",
						default=1,
						type=int,
						help="LSTM layers for the Hierarchical Model")
	parser.add_argument("--do_train",
						default=False,
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_eval",
						default=False,
						action='store_true',
						help="Whether to run eval on the dev set.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=8,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--learning_rate",
						default=5e-5,
						type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_steps",
						default=100,
						type=int,
						help="Proportion of training to perform linear learning rate warmup for. "
							 "E.g., 0.1 = 10%% of training.")
	parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
	parser.add_argument("--no_cuda",
						default=False,
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=42,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16',
						default=False,
						action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--loss_scale',
						type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	args = parser.parse_args()


	processor = dreamProcessor(args.data_dir)
	label_list = processor.get_labels()

	# num_choices = n_class
	if args.task == 'dream':
		num_choices = n_class =  3
	elif args.task == 'race':
		num_choices = n_class = 4 

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
							args.gradient_accumulation_steps))

	args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)


	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

	if not args.do_train and not args.do_eval:
		raise ValueError("At least one of `do_train` or `do_eval` must be True.")

	os.makedirs(args.output_dir, exist_ok=True)

	## only use cased model 
	tokenizer = XLNetTokenizer.from_pretrained(args.xlnet_model, do_lower_case=False) 

	train_examples = None
	num_train_steps = None
	if args.do_train:
		train_examples = processor.get_train_examples(args.data_dir)
		num_train_steps = int(
			len(train_examples) / n_class / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

	## prepare model   
	# model = XLNetForSequenceClassification.from_pretrained(args.xlnet_model,
	# 	cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
	# 	num_choices=3)
	# model.to(device)
	
	model = HCAN(args, num_choices)
	model.to(device)

	if args.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
														  output_device=args.local_rank)
	elif n_gpu > 1:
		model = torch.nn.DataParallel(model)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())

	# hack to remove pooler, which is not used
	# thus it produce None grad that break apex
	param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

	no_decay = ['bias', 'LayerNorm.weight']
	## note: no weight decay according to XLNet paper 
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	t_total = num_train_steps
	if args.local_rank != -1:
		t_total = t_total // torch.distributed.get_world_size()
	if args.fp16:
		try:
			from apex.optimizers import FP16_Optimizer
			from apex.optimizers import FusedAdam
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		optimizer = FusedAdam(optimizer_grouped_parameters,
							  lr=args.learning_rate,
							  bias_correction=False,
							  max_grad_norm=1.0)
		if args.loss_scale == 0:
			optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
		else:
			optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
	else:
		# Adam Epsilon fixed at 1e-6 according to XLNet paper 
		optimizer = AdamW(optimizer_grouped_parameters,
							lr=args.learning_rate,
							eps=args.adam_epsilon)
		warmup_steps = args.warmup_steps
		scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)


	global_step = 0
	if args.do_train:
		train_features = convert_examples_to_features(
			train_examples, label_list, args.max_sent_len, args.max_no_sent, tokenizer, n_opt=num_choices)
		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_steps)

		p_input_ids = []
		p_input_mask = []
		p_segment_ids = []
		q_input_ids = []
		q_input_mask = []
		q_segment_ids = []
		o_input_ids = []
		o_input_mask = []
		o_segment_ids = []
		label_id = []
		for f in train_features:
			p_input_ids.append([])
			p_input_mask.append([])
			p_segment_ids.append([])
			q_input_ids.append([])
			q_input_mask.append([])
			q_segment_ids.append([])
			o_input_ids.append([])
			o_input_mask.append([])
			o_segment_ids.append([])
			for i in range(num_choices):
				## put three input sequences tgt 
				# for sent in f[i].p_input_ids:
				# 	for x in sent:
				# 		if type(x) == str:
				# 			print ("Wrong type :", print (sent))
				p_input_ids[-1].append(f[i].p_input_ids)
				p_input_mask[-1].append(f[i].p_input_mask)
				p_segment_ids[-1].append(f[i].p_segment_ids)
				q_input_ids[-1].append(f[i].q_input_ids)
				q_input_mask[-1].append(f[i].q_input_mask)
				q_segment_ids[-1].append(f[i].q_segment_ids)
				o_input_ids[-1].append(f[i].o_input_ids)
				o_input_mask[-1].append(f[i].o_input_mask)
				o_segment_ids[-1].append(f[i].o_segment_ids)
			label_id.append([f[0].label_id])                

		all_p_input_ids = torch.tensor(p_input_ids, dtype=torch.long)
		all_p_input_mask = torch.tensor(p_input_mask, dtype=torch.long)
		all_p_segment_ids = torch.tensor(p_segment_ids, dtype=torch.long)
		
		all_q_input_ids = torch.tensor(q_input_ids, dtype=torch.long)
		all_q_input_mask = torch.tensor(q_input_mask, dtype=torch.long)
		all_q_segment_ids = torch.tensor(q_segment_ids, dtype=torch.long)
		
		all_o_input_ids = torch.tensor(o_input_ids, dtype=torch.long)
		all_o_input_mask = torch.tensor(o_input_mask, dtype=torch.long)
		all_o_segment_ids = torch.tensor(o_segment_ids, dtype=torch.long)
		all_label_ids = torch.tensor(label_id, dtype=torch.long)

		train_data = TensorDataset(all_p_input_ids, all_p_input_mask, all_p_segment_ids, 
			all_q_input_ids, all_q_input_mask, all_q_segment_ids, 
			all_o_input_ids, all_o_input_mask, all_o_segment_ids, all_label_ids)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_data)
		else:
			train_sampler = DistributedSampler(train_data)
		train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

		## Input Shape: P: (bsz, num_choices, max_no_sent, max_sent_len)
		## Q & O: (bsz, num_choices, max_sent_len)

		for ep in range(int(args.num_train_epochs)):
			model.train()
			max_score = 0 
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(train_dataloader):
				batch = tuple(t.to(device) for t in batch)
				p_input_ids, p_input_mask, p_segment_ids, q_input_ids, q_input_mask, q_segment_ids, o_input_ids, o_input_mask, o_segment_ids, label_ids = batch
				loss, _ = model(passage_input=(p_input_ids, p_segment_ids, p_input_mask), 
					question_input=(q_input_ids, q_segment_ids, q_input_mask), 
					option_input=(o_input_ids, o_segment_ids, o_input_mask), labels=label_ids)
				if n_gpu > 1:
					loss = loss.mean() # mean() to average on multi-gpu.
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps
				loss.backward()
				tr_loss += loss.item()
				nb_tr_examples += p_input_ids.size(0)
				nb_tr_steps += 1
				
				if (step + 1) % args.gradient_accumulation_steps == 0:
					scheduler.step()
					optimizer.step()    # We have accumulated enought gradients
					model.zero_grad()
					global_step += 1

				if step%800 == 0:
					logger.info("Training loss: {}, global step: {}".format(tr_loss/nb_tr_steps, global_step))


			if args.do_eval:
				eval_examples = processor.get_dev_examples(args.data_dir)
				eval_features = convert_examples_to_features(
					eval_examples, label_list, args.max_sent_len, args.max_no_sent, tokenizer, n_opt=num_choices)

				logger.info("***** Running Dev Evaluation *****")
				logger.info("  Num examples = %d", len(eval_examples))
				logger.info("  Batch size = %d", args.eval_batch_size)

				p_input_ids = []
				p_input_mask = []
				p_segment_ids = []
				q_input_ids = []
				q_input_mask = []
				q_segment_ids = []
				o_input_ids = []
				o_input_mask = []
				o_segment_ids = []
				label_id = []
				for f in eval_features:
					p_input_ids.append([])
					p_input_mask.append([])
					p_segment_ids.append([])
					q_input_ids.append([])
					q_input_mask.append([])
					q_segment_ids.append([])
					o_input_ids.append([])
					o_input_mask.append([])
					o_segment_ids.append([])
					for i in range(num_choices):
						## put three input sequences tgt 
						p_input_ids[-1].append(f[i].p_input_ids)
						p_input_mask[-1].append(f[i].p_input_mask)
						p_segment_ids[-1].append(f[i].p_segment_ids)
						q_input_ids[-1].append(f[i].q_input_ids)
						q_input_mask[-1].append(f[i].q_input_mask)
						q_segment_ids[-1].append(f[i].q_segment_ids)
						o_input_ids[-1].append(f[i].o_input_ids)
						o_input_mask[-1].append(f[i].o_input_mask)
						o_segment_ids[-1].append(f[i].o_segment_ids)
					label_id.append([f[0].label_id])                

				all_p_input_ids = torch.tensor(p_input_ids, dtype=torch.long)
				all_p_input_mask = torch.tensor(p_input_mask, dtype=torch.long)
				all_p_segment_ids = torch.tensor(p_segment_ids, dtype=torch.long)
				
				all_q_input_ids = torch.tensor(q_input_ids, dtype=torch.long)
				all_q_input_mask = torch.tensor(q_input_mask, dtype=torch.long)
				all_q_segment_ids = torch.tensor(q_segment_ids, dtype=torch.long)
				
				all_o_input_ids = torch.tensor(o_input_ids, dtype=torch.long)
				all_o_input_mask = torch.tensor(o_input_mask, dtype=torch.long)
				all_o_segment_ids = torch.tensor(o_segment_ids, dtype=torch.long)
				all_label_ids = torch.tensor(label_id, dtype=torch.long)

				eval_data = TensorDataset(all_p_input_ids, all_p_input_mask, all_p_segment_ids, 
					all_q_input_ids, all_q_input_mask, all_q_segment_ids, 
					all_o_input_ids, all_o_input_mask, all_o_segment_ids, all_label_ids)

				if args.local_rank == -1:
					eval_sampler = SequentialSampler(eval_data)
				else:
					eval_sampler = DistributedSampler(eval_data)
				eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

				model.eval()
				eval_loss, eval_accuracy = 0, 0
				nb_eval_steps, nb_eval_examples = 0, 0
				logits_all = []
				for p_input_ids, p_input_mask, p_segment_ids, q_input_ids, q_input_mask, q_segment_ids, o_input_ids, o_input_mask, o_segment_ids, label_ids in eval_dataloader:
					p_input_ids = p_input_ids.to(device)
					p_input_mask = p_input_mask.to(device)
					p_segment_ids = p_segment_ids.to(device)
						
					q_input_ids = q_input_ids.to(device)
					q_input_mask = q_input_mask.to(device)
					q_segment_ids = q_segment_ids.to(device)

					o_input_ids = o_input_ids.to(device)
					o_input_mask = o_input_mask.to(device)
					o_segment_ids = o_segment_ids.to(device)

					label_ids = label_ids.to(device)

					with torch.no_grad():
						tmp_eval_loss, logits = model(passage_input=(p_input_ids, p_segment_ids, p_input_mask), 
													question_input=(q_input_ids, q_segment_ids, q_input_mask), 
													option_input=(o_input_ids, o_segment_ids, o_input_mask), labels=label_ids)

					logits = logits.detach().cpu().numpy()
					label_ids = label_ids.to('cpu').numpy()
					for i in range(len(logits)):
						logits_all += [logits[i]]
					
					tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

					eval_loss += tmp_eval_loss.mean().item()
					eval_accuracy += tmp_eval_accuracy

					nb_eval_examples += p_input_ids.size(0)
					nb_eval_steps += 1

				eval_loss = eval_loss / nb_eval_steps
				eval_accuracy = eval_accuracy / nb_eval_examples

				if args.do_train:
					result = {'eval_loss': eval_loss,
							  'eval_accuracy': eval_accuracy,
							  'global_step': global_step,
							  'loss': tr_loss/nb_tr_steps}
				else:
					result = {'eval_loss': eval_loss,
							  'eval_accuracy': eval_accuracy}


				output_eval_file = os.path.join(args.output_dir, "eval_results_test.txt")
				with open(output_eval_file, "a+") as writer:
					logger.info(" Epoch: %d", (ep+1))
					logger.info("***** Eval results *****")
					writer.write(" Epoch: "+str(ep+1))
					for key in sorted(result.keys()):
						logger.info("  %s = %s", key, str(result[key]))
						writer.write("%s = %s\n" % (key, str(result[key])))
				
				# output_eval_file = os.path.join(args.output_dir, "logits_test.txt")
				# with open(output_eval_file, "w") as f:
				#     for i in range(len(logits_all)):
				#         for j in range(len(logits_all[i])):
				#             f.write(str(logits_all[i][j]))
				#             if j == len(logits_all[i])-1:
				#                 f.write("\n")
				#             else:
				#                 f.write(" ")

				if eval_accuracy > max_score:
					max_score = eval_accuracy
					## save trained model  
					model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
					output_model_file = os.path.join(args.output_dir, "pytorch_model_{}epoch.bin".format(ep+1))
					torch.save(model_to_save.state_dict(), output_model_file)
			else:
				## save trained model  
				model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
				output_model_file = os.path.join(args.output_dir, "pytorch_model_{}epoch.bin".format(ep+1))
				torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
	main()





















