# Architecture Justification: Transformer-Based Extractive QA

## A. Chosen Architecture: BERT

- BERT is the best choice as its bidirectional understanding, it can better understand long sequences
- That behavior is effective because when it understand the context, it provides accurate answers
- After searching in HugginFace is 'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad', as it is suitable for question answering tasks, after trying other architectures like 'deepset/bert-large-uncased-whole-word-masking-squad2' and testing the results

## B. Encoder only over Decoder only [ BERT vs GPT ]

- **GPT** : Use unidirectional context procesing, it processes text from left to right only, its decoder blocks predicts a token based on the previous words so it is ideal for text generation
- **BERT** : Use bidirectional processing, meaning that it reads the text from left to right and from right to left, providing deeper contextual understanding, so its encoder blocks are optimized for understanding and question answering rather than generation

## C. Self Attention Mechanism

- Self attention helps the model to learn a relationship between words, understand the context and arrangement of words in a sentence
- A traditional LSTM model handels the input words sequentially while a transformers handels them in parallel, which means that even if the answer of the question is mentioned at the end of a long sequence; a transformer can understand it and answer the question correctly, and that is due to self attention mechanism.
- So, self attention mechanism is the main mechanism that makes a transformer perform better than an LSTM, as it doesn't lose any important informtion on the long way.
- In the context of this task, he input to BERT is the question and the ticket text combined. When BERT processes this input, self attention allows question words like "component" and "system" to directly attend to relevant words in the ticket such as "Kubernetes", "firmware", or "portal". This cross attention between question and context is what enables the model to locate the correct answer span.

## D. Positional Encoding 

- Transformers process all text in parallel with self attention mechanism, so it doesn't keep the order of words, Here is where positional encoding role steps in, it adds information about the position of each token in the sequence before the text is fed to the encoder/decoder layer.
- Positional encoding helps avoiding the symmetry issues, provides contextual understanding for the text and improves generalization
- The original transformer model used a specific method based on sinusoidal functions to generate a unique position encoding for each position in the sequence to apply positional encoding, but BERT uses a different method called learned (trainable) absolute positional embeddings which is Standard for BERT family models, it is trainable and have a limit of 512 sequence lenght

## E. Why Transformers Outperform LSTMs

- LSTMs process text sequentially, which means that they read one word at a time from left to right, While they use a memory cell to carry information forward, this information gradually fades over long sequences due to the vanishing gradient problem. This means that by the time an LSTM reaches the end of a long support ticket, it may have already lost important information from the beginning of the text, making it unreliable for long-range dependency tasks.