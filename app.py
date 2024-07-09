import pandas as pd
import numpy as np
import streamlit as st
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torchmetrics

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("merged_movies_IDs.csv")


#----------------------------Header and Title----------------------------------
header = """

  Welcome to Movieta
  
   Your Ultimate Movie Recommendation System Guide 
  
"""
st.markdown(header,unsafe_allow_html=True)
#---------------------------- Load Model --------------------------------------


class recommender_model(pl.LightningModule):
  def __init__(self, total_users , total_movies ):
    super().__init__()
    self.userEmbedding = nn.Embedding(num_embeddings=total_users , embedding_dim=16)  
    self.movieEmbedding = nn.Embedding(num_embeddings=total_movies , embedding_dim=16)
    self.model_seq = nn.Sequential(
        nn.Linear(in_features=32 , out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32,out_features=1),
        nn.Sigmoid()
    )
    self.accuracy = torchmetrics.Accuracy(task="binary")

  def forward(self, users , movies):
    embedded_user = self.userEmbedding(users)
    embedded_movie = self.movieEmbedding(movies)
    vector = torch.cat([embedded_user , embedded_movie],dim=1)
    return self.model_seq(vector)
  
  def training_step(self , batch , batch_idx):
    users , movies , labels = batch
    predicted_labels = self(users,movies)
    loss = nn.BCELoss()(predicted_labels, labels.view(-1,1).float())
    self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log('train_acc_step', self.accuracy(predicted_labels,labels.view(-1,1).float()))
    return loss

#---------------------------- Prediction --------------------------------------
def prediction(interacted_movies):
  # Preparing Data
  all_rated_movies = ratings['movieId'].unique()
  # Creating temp ID for the user
  user_id = 29958
  non_interacted_movies = set(all_rated_movies) - set(interacted_movies) 
  random_non_interacted = list(set(np.random.choice(list(non_interacted_movies),1500)))
  random_non_interacted.extend(interacted_movies) 
  user_movies = random_non_interacted

  # Preducting output
  model = recommender_model(270897,176264)
  model.load_state_dict(torch.load('model2.h5'))
  predicted_labels = np.squeeze(model(torch.tensor([user_id]*len(user_movies)), torch.tensor(user_movies)).detach().numpy())
  top_50 = [user_movies[i] for i in np.argsort(predicted_labels)[::-1][0:50].tolist() if i not in interacted_movies]
  top_50_movies = []
  for i in top_50:
    try:
      i=np.int32(i).item()
      if i in interacted_movies:
        continue
      top_50_movies.append(i)
    except:
      pass
  return top_50_movies

# ----------------------Showing demo movies-------------------------------------
def title_movie(name):
   return st.markdown(f'''
   {name}
   ''', unsafe_allow_html=True)
 
def show_movies(movie_samples):
  cols = st.columns(3)
  for i in range(4):
    with st.container():
      for j in range(3):
        with cols[j]:
          movie = movie_samples.iloc[i*3+j]
          title_movie(movie['original_title'])
          
          st.image(movie['poster_path'])
          overview = movie['overview']
          if len(str(overview)) > 140:
            st.write(overview[:140])
          else:
            st.write(overview)
          st.markdown(movie['vote_average'])

tmdb_movies_name = movies['original_title'].unique()
user_list = []
top_movies=[]
def main():
  movie_samples = movies.sample(12, random_state=3)
  show_movies(movie_samples)

  options = st.multiselect("Select multiple movies ",tmdb_movies_name)
  st.write('You selected:', options)
  predict = st.button("Predict")
  if predict:
    for idx , item in enumerate(options):
      movie_id = movies[movies['original_title']==item]['movieId'].values[0]
      user_list.append(movie_id)
    top_movies = prediction(user_list)
    st.header("Movieta recommends you:")
    show_movies(movies[movies['movieId'].isin(top_movies)])
      


# --------------remove header icon and streamlit footer-------------------------
hide_default_format = """
       
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
if __name__=='__main__':
    main()    
