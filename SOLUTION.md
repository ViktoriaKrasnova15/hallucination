# My solution of Hallucination Detection

## Reproducibility Instructions
## Environment
1) Google Colab (T4 free version)
2) Requirements copied from the task repository

## Usage
Open in Colab and run in Terminal:

git clone https://github.com/YOUR_USERNAME/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION
pip install -r requirements.txt
python solution.py

## Final solution description
I've modified probe.py, aggregation.py and splitting.py.

In probe.py I've added Dropout() to the net to prevent overfitting (which was noticed during experiments).
I've also replaced Adam optimizer with AdamW optimizer with smaller lr and weight decay for the same reason.
The last alteration in this file to stop overfitting - Scheduler.

As for splitting.py, I've addend K-fold valindation on 5 folds to get more stable metric results.

In aggregation.py, I've saved the basic metric, as it has shown hight quality (over 70% ROC-AUC). 
I've also added the same approach in regards to the pre-last layer. My intuition was to look at the last layers
as they are supposed to reflect sematics. I did not look at the first layers since hallucinations and normal responses
look semantically alike and early layers reflect structure and syntax, not meaning.

I've added token variance as a metric (for the first layer, mid layer and 2 last layers). I did not count it for all of the layers to avoid overfitting.
Token variance is supposed to be lower for the normal answers in repspect to hallucinations because hallucinations are less predictable.
There are some notions of this method in scientific articles on hallucination detection. It was my attempt to include variation though through an easier approach.

I've also tried using Effective Rank-based as it was mentioned in this article https://arxiv.org/abs/2510.08389 Authors used different answers for the SVD matrix but I tried using
last 6 layers instead, since these layers are supposed to reflect meaning.

I've also used one geometric feature - cosine distance between two nearby layers. I've chosen 5 pairs of the layers so that these pairs cover the first, the last and the middle
of the model. There are some articles, whose author claim that we should look and the dynamics between the hidden states (how smooth they change), for example,
here https://aclanthology.org/2025.acl-long.880/. Counting cosine distance was my attempt to catch this cross-layer metric.

## Experiments and failed attempts

I've tried using mean pooling insted of looking at the last token hidden state, but it only worsened the quality. I've also tried to look more precisely at the hidden states
of the mid layers but it also spoiled the quality. So I decided to stick to the baseline (last token approach) and add the metrics listed above as additional ones.
