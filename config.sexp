(config
  ;;; Model architecture

  ;; Embedding dimension. Each token and position is represented as a
  ;; vector of this length. Also the width of every layer in the model.
  ;; GPT-2 small uses 768.
  (n-embd 64)

  ;; Number of transformer layers (attention + MLP blocks).
  ;; More layers = more capacity, but slower training and inference.
  ;; GPT-2 small uses 12.
  (n-layer 2)

  ;; Number of attention heads. Each head attends independently over a
  ;; slice of the embedding (head-dim = n-embd / n-head).
  ;; GPT-2 small uses 12.
  (n-head 4)

  ;; Position embedding table size. The model cannot process sequences
  ;; longer than this because there are no learned embeddings beyond it.
  ;; GPT-2 uses 1024.
  (block-size 256)

  ;;; Tokenizer

  ;; Maximum number of BPE merge rules to learn from the corpus.
  ;; More merges = larger vocabulary = each token carries more meaning.
  ;; GPT-2 uses ~50,000 merges for a 50,257-token vocabulary.
  (num-merges 256)

  ;;; Training

  ;; Number of tokens per training sequence. Should not exceed block-size.
  ;; Larger values give the model more context but use more memory.
  ;; Not a separate GPT-2 parameter; GPT-2 always trains on full
  ;; block-size (1024) sequences.
  (train-seq-len 256)

  ;; Adam optimizer peak learning rate.
  ;; Linearly decayed to zero over the course of training.
  ;; GPT-2 uses 2.5e-4 with a cosine schedule (linear warmup over 2000 steps).
  (learning-rate 0.001)

  ;; Adam first moment decay rate.
  ;; Controls the exponential moving average of gradients.
  ;; The GPT-2 paper does not specify; 0.9 is the standard Adam default.
  (beta1 0.9)

  ;; Adam second moment decay rate.
  ;; Controls the exponential moving average of squared gradients.
  ;; The GPT-2 paper does not specify; 0.999 is the standard Adam default.
  (beta2 0.999)

  ;; Adam epsilon. Small constant added to the denominator to
  ;; prevent division by zero during the parameter update.
  ;; The GPT-2 paper does not specify; 1e-8 is the standard Adam default.
  (eps-adam 1e-8)

  ;; Save a checkpoint every N training steps.
  ;; Checkpoints include weights and Adam optimizer state.
  ;; Not a GPT-2 parameter; this is an implementation detail.
  (checkpoint-interval 10)

  ;; Standard deviation of the Gaussian used to initialize weights.
  ;; Smaller values start the model closer to zero.
  ;; GPT-2 uses 0.02.
  (init-scale 0.02)

  ;;; Inference

  ;; Softmax temperature for sampling. Lower values make the model
  ;; more confident (greedy); higher values increase diversity.
  ;; Not a GPT-2 architecture parameter; temperature is a general
  ;; sampling knob chosen at inference time.
  (temperature 0.8))
