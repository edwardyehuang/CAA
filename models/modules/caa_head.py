# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
from iseg.layers import DenseExt

from iseg.utils.attention_utils import get_axial_attention
from iseg.layers.model_builder import ConvBnRelu, get_training_value

from iseg.vis.vismanager import get_visualization_manager


class ChannelizedAxialAttentionHead(tf.keras.Model):
    def __init__(
        self,
        filters=512,
        attention_blocks_num=1,
        use_channel_attention=True,
        use_image_level=True,
        num_parallel_group_fn=16,
        fallback_concat=False,
        use_entry_conv=True,
        name=None,
    ):

        if name is None:
            name = "ChannelizedAxialAttentionHead"

        super(ChannelizedAxialAttentionHead, self).__init__(name=name)

        self.use_image_level = use_image_level
        self.filters = filters

        self.entry_convbnrelu = (
            None if not use_entry_conv else ConvBnRelu(self.filters, kernel_size=(3, 3), name="entry")
        )

        self.end_convbnrelu = ConvBnRelu(self.filters, (1, 1), name="end")

        self.ca_blocks = [
            AxialAttentionBlock(
                guided_filters=64,
                filters=self.filters,
                use_channel_attention=use_channel_attention,
                num_parallel_group_fn=num_parallel_group_fn,
                fallback_concat=fallback_concat,
                name="ca_block".format(i),
            )
            for i in range(attention_blocks_num)
        ]

        if self.use_image_level:
            self.image_level_block = ImageLevelBlock(self.filters, (1, 2))

    def call(self, inputs, training=None, **kwargs):

        x = inputs

        if self.entry_convbnrelu is not None:
            x = self.entry_convbnrelu(x, training=training)

        for ca_block in self.ca_blocks:
            x = ca_block(x, training=training)

        x = [self.end_convbnrelu(x, training=training)]

        if self.use_image_level:
            x += [self.image_level_block(inputs, training=training)]

        x = tf.concat(x, axis=-1)

        return x



class ImageLevelBlock(tf.keras.Model):
    def __init__(self, filters=256, pooling_axis=(1, 2), name=None):
        super(ImageLevelBlock, self).__init__(name="ImageLevelBlock" if name is None else name)

        self.convbnrelu = ConvBnRelu(filters, (1, 1), name="conv")
        self.pooling_axis = pooling_axis

    def call(self, inputs, training=None):

        x = inputs
        inputs_dtype = inputs.dtype
        inputs_size = tf.shape(inputs)[1:3]

        x = tf.reduce_mean(x, axis=self.pooling_axis, keepdims=True, name="pool")
        x = self.convbnrelu(x, training=training)

        one = tf.ones((), dtype=tf.int32)
        target_height = tf.cast(inputs_size[0], dtype=tf.int32)
        target_width = tf.cast(inputs_size[1], dtype=tf.int32)
        target_shape = tf.stack([one, target_height, target_width, one], axis=0)

        x = tf.ones(target_shape, dtype=x.dtype) * x

        x = tf.cast(x, inputs_dtype)

        return x


class ChannelAttentionBlock(tf.keras.Model):
    def __init__(
        self,
        hiddlen_fitlers=256,
        end_filters=256,
        hidden_layers_count=1,
        hidden_activation=tf.nn.leaky_relu,
        name=None,
    ):

        super(ChannelAttentionBlock, self).__init__(name=name if name is not None else "ChannelAttentionBlock")

        self.hidden_activation = hidden_activation

        # can be replaced by 1x1 conv (same)
        self.denses = [
            DenseExt(hiddlen_fitlers, use_bias=False, name="dense{}".format(i)) for i in range(hidden_layers_count)
        ]

        self.dense_end = DenseExt(end_filters, use_bias=False, name="dense_end")

    def call(self, inputs, training=None):

        inputs_rank = len(inputs.shape)

        if inputs_rank == 4:
            x = tf.reduce_mean(inputs, axis=(1, 2))  # [N, C]
        elif inputs_rank == 3:
            x = tf.reduce_mean(inputs, axis=1)  # [N, C]
        elif inputs_rank == 2:
            x = inputs
        else:
            raise ValueError("Incorrect inputs rank")

        for i in range(len(self.denses)):
            x = self.denses[i](x)
            x = self.hidden_activation(x)

        x = self.dense_end(x)
        x = tf.nn.sigmoid(x)

        x = tf.expand_dims(x, axis=1)

        if inputs_rank == 4:
            x = tf.expand_dims(x, axis=2)

        return x


class AxialAttentionBlock(tf.keras.Model):
    def __init__(
        self,
        guided_filters=64,
        filters=512,
        hidden_layers_count=5,
        hidden_layer_ratio=0.25,
        use_channel_attention=True,
        channel_attention_norm=False,
        num_parallel_group_fn=16,
        fallback_concat=False,
        name=None,
    ):

        super(AxialAttentionBlock, self).__init__(name=name if name is not None else "AxialAttentionBlock")

        self.filters = filters
        self.use_channel_attention = use_channel_attention
        self.fallback_concat = fallback_concat
        self.num_parallel_group_fn = num_parallel_group_fn

        self.v_querykey_convbnrelu = ConvBnRelu(guided_filters, (1, 1), name="v_querykey_conv")
        self.h_querykey_convbnrelu = ConvBnRelu(guided_filters, (1, 1), name="h_querykey_conv")

        self.v_c_attention = ChannelAttentionBlock(
            int(filters * hidden_layer_ratio),
            filters,
            hidden_layers_count,
            channel_attention_norm,
            name="v_c_attention",
        )
        self.h_c_attention = ChannelAttentionBlock(
            int(filters * hidden_layer_ratio),
            filters,
            hidden_layers_count,
            channel_attention_norm,
            name="h_c_attention",
        )

        self.value_convbnrelu = ConvBnRelu(filters, (1, 1), name="value_conv")

    def call(self, inputs, training=None):

        x = inputs

        v_logits = self.compute_v_rate(x, training=training)  # [N, W, H, H]
        h_logits = self.compute_h_rate(x, training=training)  # [N, H, W, W]

        x = self.value_convbnrelu(x, training=training)

        vis_manager = get_visualization_manager()

        if vis_manager.recording:
            vis_manager.easy_add(x, name="before_x")

        x = self.compute_v_result(v_logits, x, training=training)
        x = self.compute_h_result(h_logits, x, training=training)

        if vis_manager.recording:
            vis_manager.easy_add(x, name="augmented_x")

        return x

    def compute_v_result(self, v_logits, features, training=None):

        v_logits = tf.transpose(v_logits, [2, 0, 3, 1], name="v_weights")  # [N, W, H, H] => [H, N, H, W]
        x = self.apply_attention_map(
            v_logits, features, axial_axis=1, use_channel_attention=self.use_channel_attention, training=training
        )  # [H, N, W, C]

        x = tf.transpose(x, [1, 0, 2, 3], name="v_features_result")  # [N, H, W, C]

        return x

    def compute_h_result(self, h_logits, features, training=None):

        h_logits = tf.transpose(h_logits, [2, 0, 1, 3], name="h_weights")  # [N, H, W, W] => [W, N, H, W]
        x = self.apply_attention_map(
            h_logits, features, axial_axis=2, use_channel_attention=self.use_channel_attention, training=training
        )  # [W, N, H, C]

        x = tf.transpose(x, [1, 2, 0, 3], name="h_features_result")  # [N, H, W, C]

        return x

    def compute_v_rate(self, features, training=None):

        q = k = self.v_querykey_convbnrelu(features, training=training)

        return get_axial_attention(q, k, axis=1)

    def compute_h_rate(self, features, training=None):

        q = k = self.h_querykey_convbnrelu(features, training=training)

        return get_axial_attention(q, k, axis=2)

    def apply_attention_map(self, attention_map, features, axial_axis=1, use_channel_attention=True, training=None):

        """
        attention_map : [H, N, H, W] or [W, N, H, W]
        features : [N, H, W, C]

        """

        channel_attention_func = self.v_c_attention if axial_axis == 1 else self.h_c_attention

        attention_map = tf.expand_dims(attention_map, axis=-1)  # [H, N, H, W, 1] or [W, N, H, W, 1]

        return self.group_fn(
            attention_map,
            features,
            num_group=self.num_parallel_group_fn,
            channel_attention_fn=channel_attention_func if use_channel_attention else None,
            reduce_axis=axial_axis,
            fallback_concat=self.fallback_concat,
            training=training,
        )

    def group_fn(
        self,
        attention_map,
        features,
        num_group=4,
        channel_attention_fn=None,
        reduce_axis=1,
        fallback_concat=False,
        training=None,
    ):

        # attention map [H, N, H, W, 1] or [W, N, H, W, 1]
        # features [N, H, W, C]

        features = tf.expand_dims(features, axis=0)  # [1, N, H, W, C]

        attention_map_shape = tf.shape(attention_map)
        total_length = attention_map_shape[0]

        if not get_training_value(training):
            num_group = total_length

        batch_size = attention_map_shape[1]
        height = attention_map_shape[2]
        width = attention_map_shape[3]
        channels = features.shape[-1]

        height_or_width = width if reduce_axis == 1 else height

        group_size = total_length // num_group
        group_size_remain = total_length % group_size

        padding_len = group_size - group_size_remain

        attention_map = tf.pad(
            attention_map, [[0, padding_len], [0, 0], [0, 0], [0, 0], [0, 0]], name="pad_attention_map"
        )

        pad_num_group = tf.shape(attention_map)[0] // group_size

        groups = tf.TensorArray(dtype=features.dtype, size=pad_num_group, clear_after_read=False, name="groups")

        def compuate_weighted_map(start_index, sliced_size):

            end_index = start_index + sliced_size

            sub_attention_map = attention_map[start_index:end_index]  # [group_size, N, H, W, 1]
            weighted_map = tf.raw_ops.Mul(
                x=features, y=sub_attention_map, name="weighted_mul"
            )  # [group_size, N, H, W, C]
            weighted_map = tf.reshape(
                weighted_map, [group_size * batch_size, height, width, channels]
            )  # [group_size * N, H, W, C]

            if channel_attention_fn is not None:
                # [group_size * N, H, W, C]
                weighted_map = tf.multiply(
                    weighted_map, channel_attention_fn(weighted_map, training=training), name="channel_mul"
                )

            weighted_map = tf.reduce_sum(
                weighted_map, axis=reduce_axis
            )  # [group_size * N, W, C or group_size * N, H, C]
            weighted_map = tf.reshape(
                weighted_map, [sliced_size, batch_size, height_or_width, channels]
            )  # [group_size, N, W, C or group_size, N, H, C]

            return weighted_map

        i = tf.constant(0)

        def loop_body(i, _groups):
            start_index = i * group_size
            weighted_map = compuate_weighted_map(
                start_index, group_size
            )  # [group_size, N, W, C or group_size, N, H, C]

            return tf.add(i, 1), _groups.write(i, weighted_map)

        _, groups = tf.while_loop(lambda i, _: tf.less(i, pad_num_group), loop_body, [i, groups])

        # For Apple CoreML support
        if fallback_concat:
            results = self.fallback_stack_tensor_array(groups)
            results = tf.reshape(results, [pad_num_group * group_size, batch_size, height_or_width, channels])
        else:
            results = groups.concat(name="groups_concat")

        groups.close(name="groups_close")

        results = results[:total_length]

        return results

    def fallback_stack_tensor_array(self, arr: tf.TensorArray):

        arr_size = arr.size()
        results = arr.gather(tf.range(arr_size))

        return results
