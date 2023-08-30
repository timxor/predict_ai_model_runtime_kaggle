# kaggle-predict-ai-model-runtime
kaggle-predict-ai-model-runtime



Kaggle contest 


COMPETITION TITLE: Google - Fast or Slow? Predict AI Model Runtime

https://www.kaggle.com/competitions/predict-ai-model-runtime?utm_medium=email&utm_source=gamma&utm_campaign=comp-tpugraphs-2023




data set 
size 6 GB
files 7,000

https://www.kaggle.com/competitions/predict-ai-model-runtime/data


my repo code
https://github.com/timxor/kaggle-predict-ai-model-runtime 


google-research-datasets/tpu_graphs repo
https://github.com/google-research-datasets/tpu_graphs/tree/main





https://arxiv.org/pdf/2308.13490.pdf




local repo
cd /Users/tim/code/kaggle-predict-ai-model-runtime


local dev env
cd /Users/tim/data/tpugraphs
cd /Users/tim/out










```
(tpugraphs) tim@super.macbook😀=> python tiles_evaluate.py --dirs /Users/tim/out/tpugraphs_tiles/model_fc77809d9bd45c14f688a810e9e70318
WARNING:tensorflow:From /Users/tim/Library/Python/3.9/lib/python/site-packages/tensorflow/python/ops/distributions/distribution.py:259: ReparameterizationType.__init__ (from tensorflow.python.ops.distributions.distribution) is deprecated and will be removed after 2019-01-01.
Instructions for updating:
The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.
WARNING:tensorflow:From /Users/tim/Library/Python/3.9/lib/python/site-packages/tensorflow/python/ops/distributions/bernoulli.py:165: RegisterKL.__init__ (from tensorflow.python.ops.distributions.kullback_leibler) is deprecated and will be removed after 2019-01-01.
Instructions for updating:
The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.
dataset cache file:  /Users/tim/data/tpugraphs/cache/tile/xla/9d2c87d76a2aa04cccb03e88d1a9811b-cache.npz
loaded from /Users/tim/data/tpugraphs/cache/tile/xla/9d2c87d76a2aa04cccb03e88d1a9811b-cache.npz
dataset cache file:  /Users/tim/data/tpugraphs/cache/tile/xla/64b020c278749f00b1e5703e6c5f9a75-cache.npz
loaded from /Users/tim/data/tpugraphs/cache/tile/xla/64b020c278749f00b1e5703e6c5f9a75-cache.npz
dataset cache file:  /Users/tim/data/tpugraphs/cache/tile/xla/fb70456c39b64044249d0737cbf34c47-cache.npz
loaded from /Users/tim/data/tpugraphs/cache/tile/xla/fb70456c39b64044249d0737cbf34c47-cache.npz
  0%|                                                                                | 0/1 [00:00<?, ?it/s]W0829 21:58:44.376159 8024514688 optimizer.py:70] At this time, the v2.11+ optimizer `tf.keras.optimizers.RestoredOptimizer` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.RestoredOptimizer`.
W0829 21:58:44.479212 8024514688 optimizer.py:70] At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.

100%|████████████████████████████████████████████████████████████████████| 676/676 [01:51<00:00,  6.06it/s]
100%|███████████████████████████████████████████████████████████████████████| 1/1 [01:53<00:00, 113.74s/it]
{
  "1": {
    "bert_pretraining.4x4.fp16": 0.07383066415786743,
    "inception_v3_batch_128_train": 0.07715363800525665,
    "mlperf_bert_batch_24_2x2": 0.10472889244556427,
    "resnet50.4x4.fp16": 0.1426999568939209,
    "resnet_v1_50_official_batch_128_bf16": 0.10517748445272446,
    "tf2_bert_pretrain_dynamic_batch_size": 0.06723014265298843,
    "unet_3d.4x4.bf16": 0.37605753540992737
  },
  "5": {
    "bert_pretraining.4x4.fp16": 0.01717139035463333,
    "inception_v3_batch_128_train": 0.02828279696404934,
    "mlperf_bert_batch_24_2x2": 0.04070887342095375,
    "resnet50.4x4.fp16": 0.04174179583787918,
    "resnet_v1_50_official_batch_128_bf16": 0.05139995366334915,
    "tf2_bert_pretrain_dynamic_batch_size": 0.018168827518820763,
    "unet_3d.4x4.bf16": 0.07754766941070557
  }
}
(tpugraphs) tim@super.macbook😀=> history
    1  mvn clean install -P development
    2  mvn clean install
    3  c

```












```
cd /Users/tim/data/tpugraphs
```


```
1070  python3 tiles_train.py --model=EarlyJoinSAGE --toy_data=True
 1071  pip install absl-py
 1072  pip3 install absl-py
 1073  python3 tiles_train.py --model=EarlyJoinSAGE --toy_data=True
 1074  python3 tiles_train.py --model=EarlyJoinSAGE --toy_data=True
 1075  pip3 install tensorflow tqdm tensorflow-ranking tensorflow_gnn --pre
 1076  python3 tiles_train.py --model=EarlyJoinSAGE --toy_data=True
 1077  pwd
 1078  open .
 1079  c
 1080  python tiles_train.py --model=EarlyJoinSAGE
 1081  atom /Users/tim/out/tpugraphs_tiles
 1082  c
 1083  python tiles_evaluate.py --dirs ./
 1084  python3 tiles_evaluate.py --dirs ./
 1085  python3 tiles_evaluate.py --dirs .
 1086  python3 tiles_evaluate.py
 1087  pwd
 1088  ls
 1089  c
 1090  ls -la
 1091  cd tpu_graphs
 1092  ls
 1093   -la
 1094  ls -la
 1095  pwd
 1096  cd ..
 1097  cd ..
 1098  ls -la
 1099  cd tpugraphs
 1100  ls -la
 1101  pwd
 1102  python3 tiles_evaluate.py --dirs /Users/tim/data/tpugraphs
 1103  python3 tiles_evaluate.py --dirs /Users/tim/out
 1104  python tiles_evaluate.py --dirs /Users/tim/out
 1105  python tiles_evaluate.py --dirs /Users/tim/out/tpugraphs_tiles/model_19f264c1c9ddfdf8ada61881bf824693
 1106  python tiles_evaluate.py --dirs /Users/tim/out/tpugraphs_tiles/model_fc77809d9bd45c14f688a810e9e70318
(tpugraphs) tim@super.macbook😀=> 
```





￼
















https://arxiv.org/pdf/2308.13490.pdf



