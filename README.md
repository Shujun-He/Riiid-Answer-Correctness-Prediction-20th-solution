# Riiid-Answer-Correctness-Prediction-20th-solution

Pipeline to reproduce a single 0.808/0.810 run

Group pickle file: https://www.kaggle.com/shujun717/rid-group-w-lag-time

## Architecture

I use the transformer encoder only SERT (SIngle-directional Encoder Representation from Transformers), to make predictions just one linear layer after the last encoder layer. This is probably a mistake

## Embeddings

1. Question_id
2. Prior question correctness
3. Timestamp difference between bundles
4. Prior question elapsed time
5. Prior question explanation
6. Tag cluster thanks to @spacelx 
7.  Tag vector
8. Fixed pos encoding, same as in Attention is All You Need
