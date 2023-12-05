# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2019-present NAVER Corp.
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

"""
Model OCRNet script.
"""

import torch
import torch.nn as nn
from .transformation import TPS_SpatialTransformerNetwork
from .feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from .sequence_modeling import BidirectionalLSTM
from .prediction import Attention
from nvidia_tao_pytorch.cv.ocrnet.model.fan import fan_tiny_8_p2_hybrid


class Model(nn.Module):
    """Model wrapper wrapping transformation, backbone, sequence."""

    def __init__(self, opt):
        """Init."""
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}
        # self.export = export

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel, opt.quantize)
        elif opt.FeatureExtraction == 'ResNet2X':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel, opt.quantize, no_maxpool1=True)
        elif opt.FeatureExtraction == "FAN_tiny_2X":
            self.FeatureExtraction = fan_tiny_8_p2_hybrid(in_chans=opt.input_channel, in_height=opt.imgH, in_width=opt.imgW)
            opt.output_channel = 192
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class, batch_max_length=opt.batch_max_length)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):  # pylint: disable=redefined-builtin
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            # if self.export:
            #     prediction = self.Prediction(contextual_feature.contiguous())
            #     prediction = nn.functional.softmax(prediction, dim=2)
            #     sequence_id = torch.argmax(prediction, dim=2)
            #     sequence_prob = torch.max(prediction, dim=2)
            #     return sequence_id, sequence_prob
            # else:
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction


class PostProcessor(nn.Module):
    """CTC postprocessor to convert raw ctc output to sequence_id and seqence_probablity"""

    def forward(self, prediction):
        """Forward."""
        prediction = nn.functional.softmax(prediction, dim=2)
        # sequence_id = torch.argmax(prediction, dim=2)
        sequence_prob, sequence_id = torch.max(prediction, dim=2)

        return sequence_id, sequence_prob


class ExportModel(nn.Module):
    """A wrapper class to wrap ocr model and the corresponding post process."""

    def __init__(self, ocr_model, prediction_type="CTC"):
        """Init."""
        super(ExportModel, self).__init__()
        self.ocr_model = ocr_model
        self.post_processor = PostProcessor()

    def forward(self, input, text):  # pylint: disable=redefined-builtin
        """Forward with post-process."""
        prediction = self.ocr_model(input, text, is_train=False)
        if self.post_processor is not None:
            sequence_id, sequence_prob = self.post_processor(prediction)
            return sequence_id, sequence_prob

        return prediction
