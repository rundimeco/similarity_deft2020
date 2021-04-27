# similarity_deft2020


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
