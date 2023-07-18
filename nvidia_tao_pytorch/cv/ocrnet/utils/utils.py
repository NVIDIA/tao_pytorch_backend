# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OCRNet utils module."""
import os
import pickle
import struct
import time
import tempfile
import logging
from eff.codec import encrypt_stream, decrypt_stream
import torch
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """Convert between text-label and text-index for CTC."""

    def __init__(self, character):
        """Initialize CTCLabelConverter.

        Args:
            character (str): A string containing the set of possible characters.
        """
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """Convert text-label into text-index.

        Args:
            text (list): text labels of each image. [batch_size]
            batch_max_length (int): max length of text label in the batch. 25 by default

        Return:
            text (Torch.tensor): text index for CTCLoss. [batch_size, batch_max_length]
            length (Torch.tensor): length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Args:
            text_index (numpy.ndarray): the batch of predicted text_index.
            length (list): the length of the predicted text.

        Return:
            list : the list of decoded text.
        """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """Convert between text-label and text-index for baidu warpctc."""

    def __init__(self, character):
        """Initialize CTCLabelConverterForBaiduWarpctc.

        Args:
            character (str): A string containing the set of possible characters.
        """
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """Convert text-label into text-index.

        Args:
            text (list): text labels of each image. [batch_size]

        Return:
            text (torch.Tensor): concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length (torch.Tensor): length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Args:
            text_index (numpy.ndarray): the batch of predicted text_index.
            length (list): the length of the predicted text.

        Return:
            list : the list of decoded text.
        """
        texts = []
        index = 0
        for ll in length:
            t = text_index[index:index + ll]

            char_list = []
            for i in range(ll):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += ll
        return texts


class AttnLabelConverter(object):
    """Convert between text-label and text-index for Attention"""

    def __init__(self, character):
        """Initialize AttnLabelConverter.

        Args:
            character (str): A string containing the set of possible characters.
        """
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """Convert text-label into text-index.

        Args:
            text (list): text labels of each image. [batch_size]
            batch_max_length (int): max length of text label in the batch. 25 by default

        Return:
            text (torch.Tensor): the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length (torch.Tensor): the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Args:
            text_index (numpy.ndarray): the batch of predicted text_index.
            length (list): the length of the predicted text.

        Return:
            list : the list of decoded text.
        """
        texts = []
        for index, _ in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        """init."""
        self.reset()

    def add(self, v):
        """add."""
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        """reset."""
        self.n_count = 0
        self.sum = 0

    def val(self):
        """val."""
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def validation(model, criterion, evaluation_loader, converter, opt):
    """Performs validation or evaluation of a model.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): The loss function to be used.
        evaluation_loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        converter (CTCLabelConverter): The converter for converting between text-label and text-index.
        opt (argparse.Namespace): The command-line arguments.

    Returns:
        float: The average loss over the evaluation dataset.
        float: The accuracy over the evaluation dataset.
        float: The normalized edit distance over the evaluation dataset.
        list: A list of predicted transcriptions for each sample in the evaluation dataset.
        list: A list of confidence scores for each sample in the evaluation dataset.
        list: A list of ground truth transcriptions for each sample in the evaluation dataset.
        float: The total inference time for the evaluation dataset.
        int: The total number of samples in the evaluation dataset.

    """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for _, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            if opt.baiduCTC:
                cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
            else:
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            if opt.baiduCTC:
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
            else:
                _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            # if opt.sensitive and opt.data_filtering_off:
            #     pred = pred.lower()
            #     gt = gt.lower()
            #     alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
            #     out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
            #     pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
            #     gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except Exception:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data  # pylint: disable=undefined-loop-variable


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """Encrypts an ONNX model.

    Args:
        tmp_file_name (str): The name of the temporary file containing the ONNX model.
        output_file_name (str): The name of the output file to write the encrypted ONNX model to.
        key (str): The passphrase to use for encryption.
    """
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def encrypt_pytorch(tmp_file_name, output_file_name, key):
    """Encrypts a PyTorch model.

    Args:
        tmp_file_name (str): The name of the temporary file containing the PyTorch model.
        output_file_name (str): The name of the output file to write the encrypted PyTorch model to.
        key (str): The passphrase to use for encryption.
    """
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def save_checkpoint(state, filename, key):
    """Saves a PyTorch checkpoint.

    Args:
        state (dict): The state dictionary to save.
        filename (str): The name of the output file to write the encrypted checkpoint to.
        key (str): The passphrase to use for encryption.

    """
    handle, temp_name = tempfile.mkstemp(".tlt")
    os.close(handle)
    torch.save(state, temp_name)
    encrypt_pytorch(temp_name, filename, key)
    os.remove(temp_name)


def decrypt_pytorch(input_file_name, output_file_name, key):
    """Decrypts a TAO model to a PyTorch model.

    Args:
        input_file_name (str): The name of the input file containing the encrypted TAO model.
        output_file_name (str): The name of the output file to write the decrypted PyTorch model to.
        key (str): The passphrase to use for decryption.
    """
    with open(input_file_name, "rb") as open_temp_file, open(output_file_name,
                                                             "wb") as open_encoded_file:
        decrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def load_checkpoint(model_path, key, to_cpu=False):
    """Loads a saved PyTorch checkpoint.

    Args:
        model_path (str): The path to the saved checkpoint file.
        key (str): The passphrase to use for decryption.
        to_cpu (bool, optional): Whether to load the model onto the CPU. Defaults to False.

    Returns:
        dict: The loaded state dictionary.

    """
    loc_type = torch.device('cpu') if to_cpu else None
    if model_path.endswith(".tlt"):
        handle, temp_name = tempfile.mkstemp(".pth")
        os.close(handle)
        decrypt_pytorch(model_path, temp_name, key)
        loaded_state = torch.load(temp_name, map_location=loc_type)
        os.remove(temp_name)
    else:
        loaded_state = torch.load(model_path, map_location=loc_type)
        if isinstance(loaded_state, dict):
            if "whole_model" in loaded_state:
                loaded_state = pickle.loads(loaded_state["whole_model"])

    return loaded_state


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    """Creates a logger object.

    Args:
        log_file (str, optional): The name of the log file to write to. Defaults to None.
        rank (int, optional): The rank of the process. Defaults to 0.
        log_level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The logger object.

    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
