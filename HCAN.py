import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from modeling_xlnet import XLNetModel as xlnet 
from torch.nn import CrossEntropyLoss, MSELoss


## TODO: Add Masking for paddings 

# Co-Attention LSTM
class CA_LSTM(nn.Module):
	# incorporate Question and Option as attention of Passage LSTM 
	def __init__(self, in_dim, out_dim, batch_first=True, bidirectional=True, dropoutP=0.3):
		# input shape should be batch_first 
		# p, q, o input should all have in_dim  
		super(CA_LSTM, self).__init__()
		self.trans_pq = nn.Linear(in_dim, in_dim)
		self.trans_po = nn.Linear(in_dim, in_dim)
		self.trans_qo = nn.Linear(in_dim, in_dim)
		self.map_linear = nn.Linear(4*in_dim, 4*in_dim)
		self.lstm_module = nn.LSTM(4*in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional, dropout=dropoutP)
		self.drop_module = nn.Dropout(dropoutP)

	def forward(self, p_input, q_input, o_input):
		# p_input: (bsz*no_opt, no_sent, sent_len, in_dim)
		no_sent = p_input.size(1)
		p_input = p_input.view(-1, p_input.size(2), p_input.size(3)) #(bsz*no_opt*no_sent, sent_len, in_dim)
		# q_input: (bsz*no_opt, sent_len, in_dim)
		q_input = q_input.repeat(no_sent, 1, 1) #(bsz*no_opt*no_sent, sent_len, in_dim)
		# o_input: (bsz*no_opt, sent_len, in_dim)
		# -> (bsz, no_opt*sent_len, in_dim)
		o_input = o_input.view(-1, o_input.size(1)*no_opt, o_input.size(2))
		# -> (bsz*no_sent*no_opt, sent_len, in_dim)
		o_input = o_input.repeat(no_sent, 1, 1).view(o_input.size(0)*no_sent*no_opt, -1, o_input.size(-1))

		## Q-O attention 
		trans_qo = self.trans_qo(o_input) #(bsz*no_sent*no_opt, sent_len, in_dim)
		attn_qo = p_input.bmm( torch.transpose(trans_qo, 1, 2) ) #(bsz*no_sent*no_opt, sent_len, sent_len)
		attn_qo = F.softmax(attn_qo, dim=-1)

		## Q-P attention 
		## Q-aware P representation 
		trans_pq = self.trans_pq(q_input) #(bsz*no_sent*no_opt, sent_len_q, in_dim)
		attn_pq = p_input.bmm( torch.transpose(trans_pq, 1, 2) ) #(bsz*no_sent*no_opt, sent_len_p, sent_len_q)
		attn_pq = F.softmax(attn_pq, dim=-1) 
		attn_pq_vec = attn_pq.bmm(q_input) #(bsz*no_sent*no_opt, sent_len_p, in_dim)

		## O-P attention 
		trans_po = self.trans_po(o_input) #(bsz*no_sent*no_opt, sent_len_o, in_dim)
		attn_po = p_input.bmm( torch.transpose(trans_po, 1, 2) ) #(bsz*no_sent*no_opt, sent_len_p, sent_len_o)
		attn_po = F.softmax(attn_po, dim=-1)
		attn_po_vec = attn_po.bmm(o_input) #(bsz*no_sent*no_opt, sent_len_p, in_dim)

		# (bsz*no_sent*no_opt, sent_len, 4*in_dim)
		all_con = torch.cat([p_input, attn_qo, attn_pq_vec, attn_po_vec], 2)
		p_output = self.drop_module(nn.ReLU()(self.map_linear(all_con)))

		# (bsz*no_sent*no_opt, sent_len, out_dim)
		H, _ = self.lstm_module(p_output)

		return H 


# Hierarchical Co-Attention Network 
class HCAN(nn.Module):
	def __init__(self, args, d_model, d_lstm=1024, num_choices=3):
		super(HCAN, self).__init__()
		## frozen, used for feature extraction 
		## using the same model to embed P, Q, O, may try use 3 different ones as well 
		## d_model should follow config.d_model of XLNet 
		## d_lstm can tune 
		self.sent_xlnet = xlnet.from_pretrained(args.xlnet_model,
		cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
		num_choices=num_choices)
		for param in self.sent_xlnet.parameters():
			param.requires_grad = False

		self.sent_lstm = CA_LSTM(d_model, d_lstm)
		self.para_lstm = CA_LSTM(d_model, d_lstm)

		self.proj = nn.Linear(d_lstm, 1)

		# ## for finetune 
		# self.whole_xlnet = xlnet.from_pretrained(args.xlnet_model,
		# cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
		# num_choices=3)


	def forward(self, passage_input, question_input, option_input, labels, no_opt=3):
		"""
		TODO: add a whole sequence finetune part 
		:p/q/o input: (input_ids, token_type_ids, attention_mask)
		"""
		# P shape: (bsz, no_sent, sent_len)
		p_input_ids, p_token_type_ids, p_attn_mask = passage_input 
		# Q shape: (bsz, sent_len)
		q_input_ids, q_token_type_ids, q_attn_mask = question_input 
		# Opt shape: (bsz, no_opt, sent_len); no_opt=3/4 depend on dataset 
		o_input_ids, o_token_type_ids, o_attn_mask = option_input 

		sent_len = p_input_ids.size(2)
		no_sent = p_input_ids.size(1)

		## Embedding Layer
		## use last layer output embedding for now, can also try use a linear combination of all layers  
		p_emb_outputs = self.sent_xlnet(p_input_ids.view(-1, sent_len), token_type_ids=p_token_type_ids.view(-1, sent_len), attention_mask=p_attn_mask.view(-1, sent_len))
		p_emb = p_emb_outputs[0] #(bsz*no_sent, sent_len, hid_dim)
		p_emb = p_emb.view(-1, no_sent, sent_len, p_emb.size(-1)) #(bsz, no_sent, sent_len, hid_dim)
		p_emb = p_emb.repeat(no_opt, 1, 1, 1) #(bsz*no_opt, no_sent, sent_len, hid_dim)

		q_emb_outputs = self.sent_xlnet(q_input_ids, token_type_ids=q_token_type_ids, attention_mask=q_attn_mask)
		q_emb = q_emb_outputs[0] #(bsz, sent_len, hid_dim)
		q_emb = q_emb.repeat(no_opt, 1, 1) #(bsz*no_opt, sent_len, hid_dim)

		o_emb_outputs = self.sent_xlnet(o_input_ids.view(-1, sent_len), token_type_ids=o_token_type_ids.view(-1, sent_len), attention_mask=o_attn_mask.view(-1, sent_len))
		o_emb = o_emb_outputs[0] #(bsz*no_opt, sent_len, hid_dim)
		
		## Apply CA_LSTM on every sentence of the Passage
		## output shape: (bsz*no_sent*no_opt, sent_len, d_lstm)
		sent_output = self.sent_lstm(p_emb, q_emb, o_emb)

		## We take the final time step output at sentence representation
		## Can also try average or max 
		sent_vec = sent_output[:, -1, :] #(bsz*no_sent*no_opt, 1, d_lstm)
		## no_sent is sent_len now, no_sent = 1 now 
		sent_vec = sent_vec.view(-1, 1, no_sent, sent_vec.size(-1)) #(bsz*no_opt, 1, no_sent, d_lstm)

		## (bsz*no_opt, no_sent, d_lstm)
		para_output = self.para_lstm(sent_vec, q_emb, o_emb)

		## Use max pooling to select the final representation
		## Can try other methods as well 
		para_vec, _ = para_output.max(1) #(bsz*no_opt, d_lstm)
		para_vec = para_vec.view(-1, no_opt, para_vec.size(-1)) #(bsz, no_opt, d_lstm)
		logits = self.proj(para_vec).squeeze(2) #(bsz, no_opt)
		outputs = (logits,)

		if labels is not None:
			loss_fct = 	CrossEntropyLoss()
			labels = labels.view(-1) #(bsz)
			loss = loss_fct(logits, labels)
			outputs = (loss,) + outputs

		return outputs #(loss, logits)












