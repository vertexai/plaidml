pml_py_library(
  NAME py
  SRCS
    plaidbench.py
    plaidbench/__init__.py
    plaidbench/cifar16.npy
    plaidbench/cli.py
    plaidbench/core.py
    plaidbench/frontend_keras.py
    plaidbench/frontend_ops.py
    plaidbench/gen-cifar.py
    plaidbench/gen-imdb.py
    plaidbench/golden/densenet169/infer,bs-1.npy
    plaidbench/golden/mobilenet_v2/infer,bs-1.npy
    plaidbench/golden/vgg19/infer,bs-1.npy
    plaidbench/golden/nasnet_mobile/infer,bs-1.npy
    plaidbench/golden/densenet121/infer,bs-1.npy
    plaidbench/golden/inception_v3/infer,bs-1.npy
    plaidbench/golden/xception/infer,bs-1.npy
    plaidbench/golden/resnet50/infer,bs-1.npy
    plaidbench/golden/inception_resnet_v2/infer,bs-1.npy
    plaidbench/golden/imdb_lstm/infer,bs-1.npy
    plaidbench/golden/densenet201/infer,bs-1.npy
    plaidbench/golden/vgg16/infer,bs-1.npy
    plaidbench/golden/nasnet_large/infer,bs-1.npy
    plaidbench/golden/mobilenet/infer,bs-1.npy
    plaidbench/golden/mobilenet/train,bs-1.npy
    plaidbench/imdb16.npy
    plaidbench/networks/keras/densenet121.py
    plaidbench/networks/keras/densenet169.py
    plaidbench/networks/keras/densenet201.py
    plaidbench/networks/keras/imdb_lstm.h5
    plaidbench/networks/keras/imdb_lstm.py
    plaidbench/networks/keras/inception_resnet_v2.py
    plaidbench/networks/keras/inception_v3.py
    plaidbench/networks/keras/mobilenet_v2.py
    plaidbench/networks/keras/mobilenet.py
    plaidbench/networks/keras/nasnet_large.py
    plaidbench/networks/keras/nasnet_mobile.py
    plaidbench/networks/keras/resnet50.py
    plaidbench/networks/keras/vgg16.py
    plaidbench/networks/keras/vgg19.py
    plaidbench/networks/keras/xception.py
    plaidbench/networks/ops/conv2d.py
    plaidbench/networks/ops/dense.py
    plaidbench/networks/ops/op.py
)

pml_py_wheel(
  NAME wheel
  PKG_NAME "plaidbench"
  VERSION ${PLAIDML_VERSION}
  PY_DEPS ::py
)
