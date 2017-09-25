from __future__ import print_function
import os
import random
import re
import numpy as np
import config
from big_bang_read import get_bang_convs, get_bang_ques_ans

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def basic_tokenizer(line, normalize_digits=True):
    #returns words: array of tokens
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    #removes <u>, </u>, [, ] from given line
    line = re.sub(b'<u>', b'', line) #re.sub is regex
    line = re.sub(b'</u>', b'', line)
    line = re.sub(b'\[', b'', line)
    line = re.sub(b'\]', b'', line)
    words = []
    #re.compiles a regex into a regex object so match or search can be used
    #python 3: b"" turns string into "bytes literal" which turns string into byte. Ignored in Python 2
    #r string prefix is raw string: '\n' is \,n instead of newline
    _WORD_SPLIT = re.compile(b"([.,!?\"'-<>:;)(])") #includes () for re.split below
    _DIGIT_RE = re.compile(bytes(r"\d", 'utf8'))
    #strip removes whitespace at beginning and end
    #lowercase string
    for fragment in line.strip().lower().split(): #each of these is a fragment ['you,', 'are', 'here!']
        for token in re.split(_WORD_SPLIT, fragment): #each token splits each fragment i.e. each token in ['here', '!']
            if not token: #if empty array
                continue
            if normalize_digits: #substitutes digits with #
                token = re.sub(_DIGIT_RE, b'#', token)
            words.append(token)
    return words


def load_vocab(vocab_path):
# returns words: array of words in vocab file and dictionary {word: index in vocab file}
    with open(vocab_path, 'rb') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}

def sentence2id(vocab, line):
    #for each token in line, returns word's ID in vocab or <unk>'s id if not in vocab
    return [vocab.get(token) for token in basic_tokenizer(bytes(line, 'utf8'))]

def load_data(enc_filename, dec_filename, max_training_size=None):
    #returns data_buckets: For each tuple in BUCKETS from config file, contains an array of 2 arrays: encoded ids and decoded ids
    #ex. data_buckets[0][[[36, 759, 100], [115, 225, 336]], [[27, 42, 86], [13, 350, 425]]]
    #for each bucket in config file, makes sure they are less than max enc and max dec specified in config file
    #so groups sequences with like lengths together
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'rb')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'rb')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    #array of arrays, one array for each tuple in config.Buckets
    i = 0
    #encode and decode is a line of id's from vocab
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        #encode_ids and decode_ids are arrays of ids from a line (encode and decode above)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break #break when added to appropriate bucket
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets

def _pad_input(input_, size):
    #pads by adding dimensions to make dimensions equal
    return input_ + [config.PAD_ID] * (size - len(input_))

def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    #batch major means first index of tensor is batch size
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in range(1)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    #encoder_inputs: array of padded/reversed encoded lines
    print(encoder_inputs)
    print(decoder_inputs)
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks
