"""
bend.models
===========
Model architectures used by the embedders in bend.utils.embedders.
"""


from .dilated_cnn import ConvNetForMaskedLM, ConvNetConfig, ConvNetModel
from .awd_lstm import AWDLSTMForLM, AWDLSTMConfig, AWDLSTMModelForInference