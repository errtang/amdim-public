import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import flatten, Flatten


class Evaluator(nn.Module):
    """
    Used to evaluate the encoder on global features while training.

    TODO:
    1) Support evaluator training while encoder training
    2) Support customizable classifiers

    """

    def __init__(self, n_classes, encoder_output=None, classifier_type=None, classifier_input_size=None,
                 classifier_args=None, is_conv=False, res_block=None):
        """
        Setup the classifier module. Uses MLPClassifier by default.

        :param n_classes: number of classes in the dataset
        :param encoder_output (Optional): features at the 1x1 layer
        :param classifier_input_size: input feature dimensions
        :param classifier_type: type of classifier used ex. MLP, Linear, SVM etc...
        :param classifier_args: parameters for the classifier
        :param is_conv: whether classifier will use convolutional blocks
        :param res_block: residual block architecture used for convolutional classifier
        """
        super(Evaluator, self).__init__()

        if not classifier_args:
            self.classifier_args = dict()
            self.classifier_args['n_classes'] = n_classes
        if not classifier_type:
            self.classifier_type = MLPClassifier  # default classifier
            self.classifier_args['n_hidden'] = 1024
            self.classifier_args['p'] = 0.2  # drop-out rate

        # use input size or infer input layer size from encoder output layer
        self.classifier_args['n_input'] = classifier_input_size if encoder_output is None else encoder_output.size(1)
        self.n_classes = n_classes  # separate from classifier_args in case you want to use .reset()
        self.is_conv = is_conv
        self.res_block = res_block

        if self.is_conv and self.res_block is None:
            raise Exception('Please provide a res_block architecture if you are using a convolutional classifier')

        self.classifier = self.classifier_type(**self.classifier_args)

    def forward(self, encoder_layer):
        """
        TODO: Modify to use separate classifiers
        Works as long as the following are defined for the classifier:
        1) forward(self, x)
        2) init param: n_input
        3) init param: n_classes

        NB: Always detach so gradients do not leak into encoder

        :param encoder_layer: output layer from the encoder ex. ftr_7, ftr_5, ftr_1
        :return: class logits from global features
        """

        # - always detach() -- send no grad into encoder!
        # collect features to feed into classifiers
        h_top_cls = flatten(encoder_layer).detach()
        if self.is_conv:
            h_res = self.res_block(encoder_layer.detach())
            h_res = torch.cat([h_top_cls, flatten(h_res)], dim=1)
            logits_from_classifier = self.classifier(h_res)
        else:
            logits_from_classifier = self.classifier(h_top_cls)

        return logits_from_classifier

    def reset_evaluator(self, n_classes=None):
        """
        Used to adapt the evaluator to a new encoder. Modifies the number of classes
        that a classifier has to output.

        :param n_classes: update the number of classes based on dataset
        :return: modified Evaluator
        """
        if n_classes is None:
            n_classes = self.n_classes

        return Evaluator(n_classes=n_classes,
                         classifier_input_size=self.classifier_args['n_inputs'],
                         classifier_type=self.classifier_type,
                         classifier_args=self.classifier_args)


# Define custom classifiers
# Base MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits


class BottleneckRes(nn.Module):
    '''
    ResNeXT layer with better initialization logic.
    '''
    def __init__(self, n_in, n_hid, n_out, n_groups):
        super().__init__()
        # CVPR folks are proper trolls, just set the damn params
        assert (((n_in % n_groups) == 0) and ((n_out % n_groups) == 0))
        self.relu = nn.ReLU(inplace=True)
        self.bn_a0 = nn.BatchNorm2d(n_in)
        self.conv_a1 = nn.Conv2d(n_in, n_hid, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(n_hid)
        self.conv_a2 = nn.Conv2d(n_hid, n_hid, kernel_size=3, stride=1,
                                 padding=1, bias=False, groups=n_groups)
        self.bn_a2 = nn.BatchNorm2d(n_hid)
        self.conv_a3 = nn.Conv2d(n_hid, n_out, kernel_size=1, bias=False)

    def forward(self, x):
        f_x = self.bn_a0(x)
        f_x = self.relu(f_x)
        f_x = self.conv_a1(f_x)
        f_x = self.bn_a1(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv_a2(f_x)
        f_x = self.bn_a2(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv_a3(f_x)
        return x + f_x


def res_block(n_ftr, layer):
    return nn.Sequential(
        BottleneckRes(n_ftr, (n_ftr // 4), n_ftr, 32),
        BottleneckRes(n_ftr, (n_ftr // 4), n_ftr, 32),
        BottleneckRes(n_ftr, (n_ftr // 4), n_ftr, 32),
        BottleneckRes(n_ftr, (n_ftr // 4), n_ftr, 32),
        nn.AvgPool2d(layer, stride=1, padding=0)
    )


def res_classifier(n_ftr, n_classes):
    return MLPClassifier(2 * n_ftr, n_classes, n_hidden=None, p=0.0)
