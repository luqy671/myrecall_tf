
*******************************************************************************************

sa1
SA_q = tf.nn.dropout(tf.layers.dense(item_list_add_pos, hidden_size, activation=tf.nn.tanh), keep_prob = 0.8)
sa2
SA_q = tf.math.tanh(tf.einsum('ijk,kl->ijl',item_list_add_pos, self.W1))
sa3
SA_q = item_list_add_pos

*******************************************************************************************

cand1
self.W3 = tf.get_variable("W3", [cand_num], trainable=True) #(cand_num) 
mean_eb = tf.reduce_mean(self.mid_embeddings_var, axis=0) #(embed_dim)
candidates = tf.einsum('i,j->ij',self.W3, mean_eb) #(cand_num, embed_dim)
self.candidates = tf.math.tanh(tf.einsum('ij,jk->ik',candidates, self.W1))#(cand_num, att_dim)

cand2
self.W2 = tf.get_variable("W2", [num_interest, embedding_dim], trainable=True) #(intere_num, embed_dim)
att_w = tf.einsum('ij,kj->ik',self.mid_embeddings_var, self.W2) #(item_num, intere_num)
wmean_eb = tf.einsum('ik,ij->kj',att_w, self.mid_embeddings_var) #(intere_num, embed_dim)
self.W3 = tf.get_variable("W3", [num_interest,cand_num], trainable=True) #(intere_num, cand_num)
candidates = tf.einsum('ki,kj->ij',self.W3, wmean_eb) #(cand_num, embed_dim)
self.candidates = tf.math.tanh(tf.einsum('ij,jk->ik',candidates, self.W1))#(cand_num, att_dim)

cand3
self.W2 = tf.get_variable("W2", [n_mid, cand_k], trainable=True) 
self.W3 = tf.get_variable("W3", [cand_k, cand_num], trainable=True) 
candidates = tf.einsum('jk,kc->cj',
                       tf.einsum('ij,ik->jk',self.mid_embeddings_var,self.W2),self.W3)#(cand_num, embed_dim)
self.candidates = tf.math.tanh(tf.einsum('ij,jk->ik',candidates, self.W1))#(cand_num, att_dim)