# similarity_deft2020

Method used for semantic similarity in the medical domain for the DEFT 2020 challange (Sentence Similarity : a study on similarity metrics with words and character strings, Buscaldi et al. 2020)

Needs scipy and sklearn, you can install them manually or use :
pip install -r requirements.txt

## Main function

get_distances(segments, conf, IDF) 
 - Takes a list of text segments, the first will be compared to the rest
 - Returns the distances between this segment and the others

 - Optional arguments:
   -  conf is a parameter configuration (see get_all_parameters)
   -  IDF is a list of texts (of the same domain or not) used to compute Tf-Idf


## Other functions

get_parameters()
 - returns a list of parameter values for Grid Search
 - for each token_type (word, char)
   - dist :  distance metrics ("cosine", "braycurtis", "manhattan"...)
   - pond : ponderation (None or tf-idf)
   - Nmin, Nmax : min and max token size

get_all_parameters_combinations(parameters)
  - Gives the parameter combinations in a list

grid_search(segments):
  - Given a list of segments, returns distances for each parameter combination
