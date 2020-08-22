#!/usr/bin/env python3

"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import argparse
import logging as log
import numpy as np
from openvino.inference_engine import IECore
import os
import sys
import time
from tokenizers import SentencePieceBPETokenizer


class Translator:
    """ Language translation.

    Arguments:
        model_xml (str): path to model's .xml file.
        model_bin (str): path to model's .bin file.
        tokenizer_src (str): path to src tokenizer.
        tokenizer_tgt (str): path to tgt tokenizer.
    """
    def __init__(self, model_xml, model_bin, tokenizer_src, tokenizer_tgt):
        self.model = TranslationEngine(model_xml, model_bin)
        self.max_tokens = self.model.get_max_tokens()
        self.tokenizer_src = Tokenizer(tokenizer_src, self.max_tokens)
        self.tokenizer_tgt = Tokenizer(tokenizer_tgt, self.max_tokens)

    def __call__(self, sentence, remove_repeats=True):
        """ Main translation method.

        Arguments:
            sentence (str): sentence for translate.
            remove_repeats (bool): remove repeated words.

        Returns:
            translation (str): translated sentence.
        """
        tokens = self.tokenizer_src.encode(sentence)
        assert len(tokens) == self.max_tokens, f"length of the tokens shouldn't exceed the {self.max_tokens} tokens"
        tokens = np.array(tokens).reshape(1, -1)
        translation = self.model(tokens)
        translation = self.tokenizer_tgt.decode(translation[0], remove_repeats)
        return translation


class TranslationEngine:
    """ OpenVINO engine for machine translation.

    Arguments:
        model_xml (str): path to model's .xml file.
        model_bin (str): path to model's .bin file.
        output_name (str): name of output blob of model.
    """
    def __init__(self, model_xml, model_bin):
        log.info("[ TranslationEngine ] loading network")
        log.info(f"[ TranslationEngine ] model_xml: {model_xml}")
        log.info(f"[ TranslationEngine ] model_bin: {model_bin}")
        self.ie = IECore()
        self.net = self.ie.read_network(
            model=model_xml,
            weights=model_bin
        )
        self.net_exec = self.ie.load_network(self.net, "CPU")
        self.output_name = self._get_output_name()
        assert self.output_name != "", "there is not output in model"

    def get_max_tokens(self):
        """ Get maximum number of tokens that supported by model.

        Returns:
            max_tokens (int): maximum number of tokens;
        """
        return self.net.input_info["tokens"].input_data.shape[1]

    def __call__(self, tokens):
        """ Inference method.

        Arguments:
            tokens (np.array): input sentence in tokenized format.

        Returns:
            translation (np.array): translated sentence in tokenized format.
        """
        out = self.net_exec.infer(
            inputs={"tokens": tokens}
        )
        return out[self.output_name]

    def _get_output_name(self):
        """ Get name of output blob.
        """
        output_name = ""
        for k in self.net.outputs.keys():
            if "ArgMax" in k:
                output_name = k
                break
        return output_name


class Tokenizer:
    """ Sentence tokenizer.

    Arguments:
        path (str): path to tokenizer's model folder.
        max_tokens (int): max tokens.
    """
    def __init__(self, path, max_tokens):
        log.info("[ Tokenizer ] loading tokenizer")
        log.info(f"[ Tokenizer ] path: {path}")
        log.info(f"[ Tokenizer ] max_tokens: {max_tokens}")
        self.tokenizer = SentencePieceBPETokenizer(
            os.path.join(path, "vocab.json"),
            os.path.join(path, "merges.txt")
        )
        self.max_tokens = max_tokens
        self.idx = {}
        for s in ['</s>', '<s>', '<pad>']:
            self.idx[s] = self.tokenizer.token_to_id(s)

    def encode(self, sentence):
        """ Encode method for sentence.

        Arguments:
            sentence (str): sentence.

        Returns:
            tokens (np.array): encoded sentence in tokneized format.
        """
        tokens = self.tokenizer.encode(sentence).ids
        tokens = self._extend_tokens(tokens)
        return tokens

    def decode(self, tokens, remove_repeats=True):
        """ Decode method for tokens.

        Arguments:
            tokens (np.array): sentence in tokenized format.
            remove_repeats (bool): remove repeated words.

        Returns:
            sentence (str): output sentence.
        """
        sentence = self.tokenizer.decode(tokens)
        for s in self.idx.keys():
            sentence = sentence.replace(s, '')
        if remove_repeats:
            sentence = self._remove_repeats(sentence)
        return sentence.lstrip()

    def _extend_tokens(self, tokens):
        """ Extend tokens.

        Arguments:
            tokens (np.array): sentence in tokenized format.

        Returns:
            tokens (np.array): extended tokens.
        """
        tokens = [self.idx['<s>']] + tokens + [self.idx['</s>']]
        pad_length = self.max_tokens - len(tokens)
        if pad_length > 0:
            tokens = tokens + [self.idx['<pad>'] for i in range(pad_length)]
        return tokens

    def _remove_repeats(self, sentence):
        """ Remove repeated words.

        Arguments:
            sentence (str): sentence.

        Returns:
            sentence (str): sentence in lowercase without repeated words.
        """
        tokens = sentence.lower().split()
        tokens_new = []
        if len(tokens):
            tokens_new = [tokens[0]]
            for i in range(1, len(tokens)):
                if tokens[i] != tokens_new[-1]:
                    tokens_new.append(tokens[i])
        return " ".join(tokens_new)


def build_argparser():
    """ Build argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-xml', type=str, required=True,
                        help='Required. Path to an .xml file with a trained model.')
    parser.add_argument('--model-bin', type=str, required=True,
                        help='Required. Path to an .bin file with a trained model.')
    parser.add_argument('--tokenizer-src', type=str, required=True,
                        help='Required. Path to the folder with src tokenizer that contains vocab.json and merges.txt.')
    parser.add_argument('--tokenizer-tgt', type=str, required=True,
                        help='Required. Path to the folder with tgt tokenizer that contains vocab.json and merges.txt.')
    return parser


def main(args):
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    log.info("creating translator")
    model = Translator(**vars(args))
    log.info("enter 'q!' to exit.")
    while True:
        sentence = input("> ")
        if sentence == "q!":
            break
        try:
            start = time.time()
            translation = model(sentence)
            stop = time.time()
            log.info(translation)
            log.info(f"time: {stop - start} s.")
        except Exception as e:
            log.error(str(e))


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
