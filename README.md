# Vietnamese Sentiment Graph for Structured Sentiment Analysis   

The source code is referenced from: https://github.com/jerbarnes/sentiment_graphs     

Dataset: ViTOED (to be updated)    

Authors: Chanh Vo, Son T. Luu, Ngan Nguyen     

Publication: To be announced. 

# Contact information: 
Son T. Luu (Email: sonlt@uit.edu.vn)
Alternative: son.lt1103@gmail.com.  

# How to run 
**Step 0**: 
+ Preparing dataset. Put the dataset (.json file) in the "data/vietnam" folder
+ Preparing the word vectors for Vietnamese: Word2VEC (http://vectors.nlpl.eu/repository/) or fasttext (https://fasttext.cc/docs/en/crawl-vectors.html). Put the vector in the **"vectors"** folders. 

**Step 1**: Converting the dataset into CoNLLU
Run the bash script "./data/create_vietnam_sent_graphs.sh"      

**Step 2**: Constructing BERT contextualize embedding.    
Run the bash script "do_embedding.sh"       

**Step 3**: Run the baseline.  
Run the bash script "./script/run_bert.sh"

**Step 4**: Run the evaluation.  
Run the bash script ".eval.sh"

## Note: 
+ If you want to change the BERT embedding, please change in the file "./script/run_bert.sh". 
+ If you want to change the word embedding vectors, please change in the file "./script/run_sentgraph_bert.sh" (line 25-26).
+ If you want to change the word embedding dim and other, please change in the file "./config/sgraph_bert.cfg" (section [network_size])



