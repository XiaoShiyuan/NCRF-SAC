# NCRF-SAC
Code for EMNLP-IJCNLP 2019 accepted paper "Similarity Based Auxiliary Classifier for Named Entity Recognition"

Our code is based on https://github.com/guillaumegenthial/tf_ner, a very good implemention. You can find data preprocessing instruction in their git.

To use elmo, clone https://github.com/allenai/bilm-tf.git and put it in model/NCRFSAC/. Then download pretrained elmo model and configuration from https://allennlp.org/elmo. You need to put these two files in model/NCRFSAC/weights/. (if this folder is not existed, create this folder) 

Notice that you need to apply for ontonotes 5.0 dataset. You could find preprocessing scripts online.

We recommend iobes tag scheme for conll2003 dataset and iob tag scheme for ontonotes.

The best lambda for iobes tag scheme is 0.05 and 0.5 for iob tag scheme

We think that iobes are suitable for datasets with fewer entity types, while iob is suitable for datasets with more entity types.

required: tensorflow >= 1.11(we test 1.11, 1.12)

Steps:

1. Download glove(840B, 300d) and put it in data/conll/. Then run build_glove.py
2. Run train_conll.sh or train_ontonotes.sh
3. Run get_best_10.py to get 10 best checkpoint
4. Modify results/model/checkpoint with top1 checkpoint returned by step 2(notice that we save latest 50 checkpoints, so you should choose the best one from these 50 checkpoints. Some times the second one or the third one is better than the first one)
5. Run predict_conll.sh or predict_ontonotes.sh
6. Run "perl conlleval < results/score/testb.preds.txt"

