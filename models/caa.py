# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import iseg.static_strings as ss
import tensorflow as tf

from iseg.backbones.feature_extractor import get_backbone
from iseg.utils import resize_image
from iseg.layers.model_builder import ConvBnRelu

from iseg import SegFoundation
from models.modules.caa_head import ChannelizedAxialAttentionHead

# CAA was implemented in 2020, long time ago
# SegFoundation will be replaced by segmanaged in future (after I have time)
class CAA(SegFoundation):
    def __init__(
        self,
        backbone_name=ss.RESNET50,
        backbone_weights_path=None,
        num_class=2,
        output_stride=16,
        use_channel_attention=True,
        use_image_level=True,
        use_aux_loss=True,
        aux_loss_rate=0.4,
        use_typical_aux_feature=False,
        always_map_fn=False,
        num_parallel_group_fn=16,
        **kwargs,
    ):

        super().__init__(
            num_class=num_class,
            num_aux_loss=1 if use_aux_loss else 0,
            aux_loss_rate=aux_loss_rate,
        )

        self.output_stride = output_stride
        self.use_aux_loss = use_aux_loss
        self.use_image_level = use_image_level
        self.use_typical_aux_feature = use_typical_aux_feature

        self.backbone = get_backbone(
            backbone_name,
            output_stride=output_stride,
            resnet_multi_grids=[1, 2, 4],
            resnet_slim=True,
            weights_path=backbone_weights_path,
            return_endpoints=True,
        )


        self.seg_head = ChannelizedAxialAttentionHead(
            filters=512,
            use_channel_attention=use_channel_attention,
            use_image_level=self.use_image_level,
            num_parallel_group_fn=num_parallel_group_fn,
            fallback_concat=always_map_fn,
        )


        self.seg_head_convbnrelu = ConvBnRelu(
            256, (1, 1), dropout_rate=0.1, name="seg_head_conv"
        )

        self.logits_conv = tf.keras.layers.Conv2D(num_class, (1, 1), name="logits_conv")

        if self.use_aux_loss:
            aux_conv_name = "aux_feature_conv0"

            if self.use_typical_aux_feature:
                aux_conv_name = "aux_feature_typical_conv0"
                self.aux_down_conv = ConvBnRelu(512, (3, 3), name="aux_down_conv")

            self.aux_feature_convbnrelu0 = ConvBnRelu(256, (1, 1), dropout_rate=0.1, name=aux_conv_name)
            self.aux_loss_logits_conv = tf.keras.layers.Conv2D(num_class, (1, 1), name="aux_loss_logits_conv")

    def call(self, inputs, training=None, **kwargs):

        input_shape = tf.shape(inputs)
        input_size = input_shape[1:3]

        x = inputs

        endpoints = self.backbone(x, training=training, **kwargs)
        endpoints = endpoints[1:]

        if self.use_aux_loss:
            # aux_layer_index = -1 if not self.use_typical_aux_feature else -2
            aux_layer_feature = endpoints[-1]  # self.aux_layer()

            if self.use_typical_aux_feature:
                aux_layer_feature = self.aux_down_conv(aux_layer_feature, training=training)

        x = endpoints[-1]
        x = self.seg_head(x, training=training)
        x = self.seg_head_convbnrelu(x, training=training)
        x = self.logits_conv(x)

        x = resize_image(x, size=input_size)
        x = tf.cast(x, tf.float32)

        if self.use_aux_loss:
            aux_feature = self.aux_feature_convbnrelu0(aux_layer_feature, training=training)
            aux_feature = self.aux_loss_logits_conv(aux_feature)
            aux_feature = resize_image(aux_feature, size=input_size)

            aux_feature = tf.cast(aux_feature, tf.float32)
            return [x, aux_feature]

        return x
