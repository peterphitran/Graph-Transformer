# Notes

https://colab.research.google.com/drive/1jw6YKLhKPY1ExdW1mGor1rxQT1VRwzuG#scrollTo=LlWWga8ixwvg

## Transformers

- deep neural network architectures that operate on multisets of input tokens
- use of multi-head attention mechanism
- positional encodings encode the underlying sequential structure of the tokens

Multi-head attention mechanism
- feature embedding considers all other tokens feature embeddings, attention to every word is considered, with different attention heads 
- attention mechanism is invariant to order of inputs (does not know order of inputs but words insentences have a natural order)
- to tackle this we use positional encodings
- transformers dont see a sequence but a multi set of tokens (ex: {This, is, a, transformer})
- Sequential order of input tokens is injected through positional encodings

Advantages of Transformer over recurrent neural networks
- Non-local dependencies: self-attention allows capturing dependencies across all input tokens
- Parallelization & Scalability: can operate over all tokens in the input at one time, unlike RNNs (Recurrent Neural Network)
- Interpretability: attention mechanism allows to investigate implicit dependencies between tokens

## Graph Neural Networks
- NN architectures that lean features of nodes in a graph in an end-to-end manner
- Learn feature representations of nodes in a graph by aggregating local neighbors' features through the use of message-passing
- Attention mechanism can be constructed as message passing within a graph
- Positional encodings are functions of the input sequential structure

A graph view of positional encoding in Text-Transformer:
- 1. Consider line graph adjacency matrix
- 2. Compute Laplacian
- 3. Extract eigenvectors of Laplacian, U
- 4. Select k non-trivial eigenvectors sorted by eigenvectors
- 5. Get the final positional encodings from the selected eigenvectors, which is [N,k]

## Graph Transformers
Extension of the standard Transformer architecture from sequences to relational and graph-structured data
- they allow better node identifiability 
- attention across all or many nodes, potentiall capturing broader structural context

Generalization of Transformers to Graphs
- 1. message passing aggregation with the use of attention function
- 2. nodes are injected with laplacian positional encodings - defined by graph structure

nodes in graphs lack canonical order, so we need a way to represent node positions in graph structure
- what we desire graph PEs should be distance-aware, unique and generalizable
- help to disambiguate nodes

Random walk PE - encodes the probability of a random walk starting from a node i visit to itself in k steps
- each node gets a unique vector as long as its k-hop neighborhood is unique

Coverage of attention
- while local attention to neighbors is useful, it may not provide broader structural context
- some tasks may benefit from aggregating messages from distant nodes
- only using local message-passing can lead to information loss
- global or required attention to distant nodes alleviates the issue
- both local and global attention ehance GT 

epoch is one complete pass through the entire training dataset
training loss for epoch, lower values mean more accurate predictions 
val mean absolute error calculated on validation dataset, lower validation means better generalization to new data helping to detect overfitting
test mean absolute error based on test dataset, thus models ability to perform on new unseen molecular graphs