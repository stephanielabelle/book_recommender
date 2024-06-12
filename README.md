# Book Recommendation Model

### Objective
Utilize K-Nearest Neighbor model to produce book recommendations based on an input book-title.  

### Data
Link: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Ratings.csv

### Cleaning of Dataset
[Books.csv](Resources/Books.csv) was 'left' merged with [Ratings.csv](Resources/Ratings.csv) and all ratings of 0 were filled with NaN.  Book-Author and Book-Title column were cleaned by consolidating name variations using regex (notebook: [book_recommender.ipynb](book_recommender.ipynb)).  
### Filtering of Dataset
The User-IDs with less than 15 books purchased, and books with less than 50 purchases were removed from the dataset.  These cutoff points were utilized to achieve a target sparsity of <99.5%.  

### K-Nearest Neighbors Model
The code of the KNN model is availabe in jupyter notebook file [book_recommender.ipynb](book_recommender.ipynb).
The cleaned and filtered dataset was converted into a pivot table and then compressed into a sparse row matrix with SciPy.
A Scikit-learn K-Nearest Neighbors model was intitiated with a cosine measurement metric and brute algorithm.  The recommender model returns 5 book recommendations and 5 author recommendations.  The author recommendations are based on the return of nearest neighbor books.  The recommender model output was tested by querying the original dataset.  Heatmaps were constructed that visualize how individual users rate the recommended books.  Assessment users were selected based on a high rating of the model input book and their total number of ratings of the recommended books.  The KNN model assessment revealed fairly accurate prediction of book satisfaction for a number of users.  
