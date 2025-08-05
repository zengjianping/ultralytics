#!/bin/bash

MODEL_DIR=datas/models
MODEL_NAME=yolo11s
USE_FP16="--fp16" # FP16模型推理更快，可能模型效果会有一点点降低，如果不使用FP16模型则用#注释掉此行

MODEL_BATCH=1
MODEL_SIZE=640x640
CUSTOM_PLUGIN="lib/plugin/libcustom_plugins.so"

EXTRA_OPTS=""
EXTRA_OPTS+=" --minShapes=images:1x3x${MODEL_SIZE} --optShapes=images:${MODEL_BATCH}x3x${MODEL_SIZE} --maxShapes=images:${MODEL_BATCH}x3x${MODEL_SIZE} $EXTRA_OPTS"
#EXTRA_OPTS+=" --staticPlugins=$CUSTOM_PLUGIN --setPluginsToSerialize=$CUSTOM_PLUGIN"

trtexec $USE_FP16 $EXTRA_OPTS \
    --onnx=$MODEL_DIR/$MODEL_NAME.onnx \
    --saveEngine=$MODEL_DIR/$MODEL_NAME.engine

