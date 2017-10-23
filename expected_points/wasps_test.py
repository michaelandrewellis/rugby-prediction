import model
import pickle

bookie_spread=8.5
ratings=pickle.load(open('ratings.pickle','rb'))
spread,p,data = model.prediction('Wasps','Harlequins',n_sims = 1000,use_ratings = ratings,spread_data = 1)
print(spread,p,sum(spread >= bookie_spread  for spread in data)/len(data))