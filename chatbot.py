from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel
import config
import data
import webbrowser
import re
from flask import Flask, render_template, request

def _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))

def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")


def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])

def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    #print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

def chat_app():
    app = Flask(__name__)
    conv = []
    @app.route('/')
    def load_page():
        return render_template('index.html', chat_output=['Hi, I am looking forward to talking to you'])
    _, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Hi, looking forward to talking to you. Press enter to leave. Max length is', max_length)
        @app.route('/comments/', methods=['GET', 'POST'])
        def foo():
            if request.form['submit'] == 'Clear':
                del conv[:]
                return render_template('index.html', chat_output=['Hi, I am looking forward to talking to you'])
            line = request.form['message']
            conv.append("Human: " + line)
            token_ids = data.sentence2id(enc_vocab, str(line))
            bucket_id = _find_right_bucket(len(token_ids))
            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])], 
                                                                            bucket_id,
                                                                            batch_size=1)
            if (len(token_ids) > max_length):
                response = "Exceeded maximum length of " + str(max_length)
            elif line.find('email') is not -1:
                response = construct_email(line)
            else:
                _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
                response = _construct_response(output_logits, inv_dec_vocab)
            conv.append('Bot: ' + response)
            return render_template('index.html', chat_output=conv)
        app.run(host='0.0.0.0')

def construct_email(line):
    recipients = ['username@company.com']
    message = "Test"
    if line.find('@') is not -1:
        recipients = re.findall('\w+\@\w+.\w+', line)
    if line.find(':') is not -1:
        message = line[line.find(':') + 1:]
    webbrowser.open("mailto:%s?subject=%s&body=%s" % (','.join(recipients), 'Subject', message))
    return 'Please add a Subject to email before sending.'

def main():
    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepare_raw_data()
        data.process_data()

if __name__ == '__main__':
    main()
