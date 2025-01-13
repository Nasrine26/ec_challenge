# Drug-Disease Link Prediction with Knowledge Graph Embeddings

## Overview
This project tackles a biological knowledge graph challenge: predicting drug-disease relationships. The dataset consists of nodes representing drugs, diseases, and other biomedical entities (gene), and edges representing their relationships. The primary goal is to train and evaluate a classifier to predict drug-disease links.

## How to run
1. Works with Python 3.10.7 and should work with Python 3.10
2. Ensure all dependencies are installed by creating a virtual env called `ec_challenge` in `/path/to/env/`:
   ```
   python -m venv /path/to/env/ec_challenge
   source /path/to/env/ec_challenge/bin/activate
   pip install networkx[default]
   pip3 install -U scikit-learn
   pip install pandas
   pip install ipykernel
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install torch_geometric
   pip3 install jupyter
   python -m ipykernel install --user --name ec_challenge
   ```
3. Download data from [here](https://drive.google.com/drive/folders/1swCsdUeYnMYLIEKYZ5ed1YIU0X1vyt9u)
4. Place the data files in the `./data/` directory:
   - `Edges.csv`
   - `Nodes.csv`
   - `Ground Truth.csv`
   - `Embeddings.csv`
5. Run notebook `link_prediction_RF.ipynb`

## Approach

### Data Overview
The provided dataset consists of:
- **Edges.csv**: Describes relationships between nodes.
- **Nodes.csv**: Contains metadata about nodes.
- **Ground Truth.csv**: Specifies drug-disease pairs with binary labels (`1` for a known link, `0` if no link).
- **Embeddings.csv**: Precomputed topological embeddings for each node.

### Pipeline
1. **Data Exploration and Preparation**:
   - The datasets were loaded and explored for structure and insights.
   - Based on the initial exploration, the data include multiple types of entities (e.g., drugs, diseases, genes) and relationships (e.g., "biolink:treats", "biolink:same_as"). A **heterogeneous graph** would be a natural choice as it explicitly models these contextual differences. 
   - Therefore, I model the data as a heterogeneous graph.
   
2. **Graph representation**:
   - The knowledge graph was represented using PyTorch Geometric's `HeteroData` object:
       - **Nodes**: Each node was mapped to unique indices.
       - **Edges**: Relationships between nodes were encoded in a sparse adjacency matrix using `edge_index`.
   - The graph was modeled as undirected.

3. **Combining Drug-Disease Node Embeddings**:
   - Pre-computed node embeddings were converted from strings to NumPy arrays.
   - A dictionary was created to map each node ID to its embedding.
   - Drug-disease pairs in the `Ground Truth.csv` were represented using their node embeddings.
   - A **Hadamard product** (element-wise multiplication) was chosen to combine node embeddings for source-target pairs to capture relational patterns.

4. **Class Balancing**:
   - The classes are moderately unbalanced - I tracked class-specific metrics such as precision, recall, and F1-score.
   - Stratified splits were used to evenly distribute the values of `y` in each split across training, validation, and test sets (70%, 15%, 15%).

5. **Model Training and Hyperparameter Tuning**:
   - A **Random Forest Classifier** was selected.
   - Hyperparameters were tuned using `RandomizedSearchCV` with 5-fold cross-validation to optimize for **ROC AUC**.

6. **Final Model**:
   - The best hyperparameters were used to retrain the model on combined training and validation data.
   - The model was evaluated on an unseen test set.

### Key Design Choices
1. **Heterogeneous Graph Representation**:
   - The data include multiple types of entities (e.g., drugs, diseases, genes) and relationships (e.g., "drug-treats-disease," "gene-translates_to-protein")
   - Undirected graph choice. While some relationships are inherently directional (e.g., "biolink:treats", "biolink:prevents", "biolink:causes") are directed, others (e.g., "biolink:same_as", "biolink:gene_associated_with_condition") are undirected. Using an undirected graph allows the model to propagate information bidirectionally across all relationships, simplifying the architecture - but I may lose key features of the graph.

2. **Embedding Combination Method**:
   - The Hadamard product was used as it preserves relational properties and is computationally efficient.
   - Alternative methods like concatenation were considered.

3. **Classifier Choice**:
   - Random Forest was chosen. Here, we care about predicting drug-disease pair links. For simplicity and speed, I selected a Random Forest Classifier.
   - Class weighting was applied to handle imbalance (`class_weight='balanced'`).

4. **Metrics**:
   - **ROC AUC**: Metric for evaluating probabilistic predictions.
   - **Precision-Recall AUC**: Assesses performance of classifier for class prediction.
   - **Normalized Confusion Matrix**: Displays class-specific prediction proportions normalized by true class size.
     
## Challenges and Limitations

1. **Graph Assumed Undirected**:
   - While some relationships (e.g., "biolink:treats", "biolink:prevents", "biolink:causes") are directed, others (e.g., "biolink:same_as", "biolink:gene_associated_with_condition") are undirected.
   - Using an undirected graph simplifies the model by ensuring bidirectional message passing, but it may lead to information loss for directed relationships (e.g., "biolink:treats").
   
2. **Precomputed Node Embeddings**:

    - This current implementation uses precomputed node embeddings. This limits the ability of the model to learn drug-disease-specific embeddings from the graph structure and relationships. Also are the pre-computed embeddings generated on the whole graph? This may lead to potential data leakage... see Future Work section for suggestions to generate node embeddings.

## Future Work
I could extend this work as following:

- Leveraging the heterogeneous graph to generate node embeddings using a graph neural network (GNN) framework with `torch_geometric.nn.HeteroConv` layers or use an encoder to generate node embeddings on the whole graph.
- I would split the heterogeneous graph into training and testing edges based on the types of edges (drug-disease links) for classifier link prediction. But which ones to pick? It is unclear how to chose the `["Drug", RELATIONSHIP, "Disease"].edge_label` for drug-disease links prediction. Should I use "biolink:treats", or all links with drugs and diseases.

---
