import math
import random
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_parameters():
  """
  A list of parameter values for Grid Search
  for each token_type (word, char)
    dist :  distance metrics
    pond : ponderation (None or tf-idf)
    Nmin, Nmax : min and max token size
  """
  l_dist = ["cosine", "braycurtis", "manhattan", "minkowski", "euclidean", "jaccard", "dice"]
  parameters ={ 
  "word":{
    "dist":l_dist,
    "pond": [None, "tf-idf"],
    "Nmin": range(1,5),
    "Nmax": range(1,5)},
  "char":{
    "dist":l_dist,
    "pond": [None],
    "Nmin": range(1,11),
    "Nmax": range(1,16)},
  }
  return parameters

def get_all_parameters_combinations(parameters):
  """
  Gives the parameter combinations in appropriate format
  """
  P = []
  for typTok, p in parameters.items():
    for Tdist in p["dist"]: 
      for pond in p["pond"]:
        ngram_ranges = [[m, M] for m in p["Nmin"] for M in p["Nmax"] if m<=M]
        for n_r in ngram_ranges:
          P.append({"Tokens":typTok, "pond":pond, "Ngrams": n_r, "dist":Tdist})
  return P

def grid_search(segments):
  """
  Given a list of segments, returns distances for each parameter combination
  """
  results = []
  parameters = get_parameters()
  parameters_combi = get_all_parameters_combinations(parameters)
  for conf in parameters_combi:
    D = get_distances(segments, conf)
    results.append([conf, D])
  return results

def get_distances(segments, conf={"dist":"cosine", "pond":None,"Tokens":"word","Ngrams":(1,1)}, IDF = []):
  """
  Takes a list of texts, the first will be compared to the rest
  Returns the distances between this segment and the others
  IDF is a list of texts (of the same domain or not) used to compute Tf-Idf
  """
  N = conf["Ngrams"]
  if conf["pond"]==None:
    V = CountVectorizer(ngram_range=N, analyzer=conf["Tokens"])
  elif conf["pond"]=="tf-idf":
    V = TfidfVectorizer(ngram_range = N)
    V.fit(raw_documents= IDF)
  X = V.fit_transform(segments).toarray()
  distances = []
  for x in X[1:]:
    if conf["dist"]=="cosine":
      dist = spatial.distance.cosine(X[0], x)
    elif conf["dist"]=="euclidean":
      dist = spatial.distance.euclidean(X[0], x)
    elif conf["dist"]=="dice":
      dist = spatial.distance.dice(X[0], x)
    elif conf["dist"]=="jaccard":
      dist = spatial.distance.jaccard(X[0], x)
    elif conf["dist"]=="minkowski":
      dist = spatial.distance.minkowski(X[0], x)
    elif conf["dist"]=="manhattan":
      dist = spatial.distance.cityblock(X[0], x)
    elif conf["dist"]=="braycurtis":
      dist = spatial.distance.braycurtis(X[0], x)
    distances.append(float(dist))
  distances = [x if math.isnan(x)==False else 1 for x in distances]
  return distances

if __name__=="__main__":
  segments = ['Députation du clergé', 'Assemblée du Palais-Royal', 'Deuxième déclaration d’amnistie', 'Députation de six corps de marchands']
  parameters = get_parameters()
  combinations = get_all_parameters_combinations(parameters)

  # Let's pick a parameter configuration randomly
  rand = random.randint(0, len(combinations))
  conf = combinations[rand]
  print(conf)
  D = get_distances(segments, conf)
  print(segments[0])
  for i in range(1, len(segments)):
    print("  dist=%f VS %s"%(round(D[i-1],4), segments[i]))
