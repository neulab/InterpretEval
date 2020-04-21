import os
import pickle

emb_dir = "preComputed/emb-ner/"
vocab_emb_fns = os.listdir(emb_dir)
for vocab_emb_fn in vocab_emb_fns:
	corpus_type = vocab_emb_fn.split('_')[0]
	print('corpus_type: ',corpus_type)
	vocab_emb_fn =emb_dir + vocab_emb_fn
	fread = open(vocab_emb_fn, 'rb')
	unique_words, emb_vecs = pickle.load(fread)

	fn_vocab_write = emb_dir+corpus_type+'_vocab.txt'
	fwrite = open(fn_vocab_write, 'wb')
	pickle.dump(unique_words, fwrite)
	fwrite.close()



