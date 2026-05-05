Movie Recommendation System: A Comparative Study

AI and Modeling Project
This project explores three different machine learning architectures to provide personalized movie recommendations using the MovieLens dataset. We compare traditional collaborative filtering with graph-based and deep learning-based approaches.

Overview
The system processes user-movie interaction data to predict which movies a user is likely to enjoy. We evaluate the performance of three models:
   1.	Matrix Factorization (MF): A baseline model using dot-product embeddings.
   2.	Graph Convolutional Network (GCN): A graph-based approach that captures high-order  neighborhoods in a bipartite user-movie graph.
   3.	Neural Collaborative Filtering (NCF): A deep learning model combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptrons (MLP).

Prerequisites
Before running the script, ensure you have the following installed:
•	Python 3.8+
•	PyTorch
•	PyTorch Geometric (PyG)
•	Pandas
•	NumPy
•	NetworkX
•	Matplotlib
You can install the dependencies via pip:
   pip install torch torch-geometric pandas numpy networkx matplotlib

Dataset Requirements
To execute the code in VS Code, ensure the following two files are present in the same directory as your script:
   •	movies.csv: Contains movieId and title.
   •	ratings.csv: Contains userId, movieId, and rating.
Note: These are standard MovieLens dataset files. They can be downloaded from https://www.kaggle.com/datasets/mexwell/fakereviews-dataset/data

How to Run
   1.	Open the project folder in VS Code.
   2.	Ensure movies.csv and ratings.csv are in the folder.
   3.	Run the script:
      python your_script_name.py
   4.	Interaction: The script will ask you to enter a userId from the dataset.
   5.	Output: * The script will display dataset statistics (Density, Sparsity, etc.).
      o	A Bipartite Graph visualization will pop up.
      o	The models will begin training (20 epochs each).
      o	Finally, you will see a comparison table of Recall@10 and specific movie recommendations for the chosen user.

Model Architectures
   1. Matrix Factorization (MF)
      Maps users and movies into a shared latent space. The recommendation score is calculated as the inner product of user and movie vectors:
         Score = u_{idx} dot m_{idx}
   2. GCN Recommender
      Utilizes PyTorch Geometric to treat the recommendation task as a link prediction problem on a graph. It propagates information across edges to refine embeddings based on the "friend of a friend" logic.
   3. Neural Collaborative Filtering (NCF)
      A hybrid architecture that fuses a linear GMF layer with a non-linear MLP. This allows the model to capture both low-rank and high-dimensional patterns in user behavior.

Evaluation Metric
We use Recall@10 to measure performance. This represents the proportion of movies the user actually liked (in the test set) that appeared in the model's top 10 recommendations.

Contributors
   •	Lakshmi Ramanathan
   •	Lakshmi Katravulapalli

