class Model_Mine(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_Mine, self).__init__(n_mid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="ComiRec_SA")
        
        #-------------------------------------------------------------------------------------------
        print('*'*25 + 'this is my model' + '*'*25)
        f_mycand = True;
        f_trans = True;
        cand_num = 200
        
        att_dim = 4 * hidden_size
        self.W1 = tf.get_variable("W1", [embedding_dim, att_dim], trainable=True) #(embed_dim, att_dim)
        
        if f_mycand:
            print('*'*25+'candidate genarate from eb when W3[cand_num]'+'*'*25)
            self.W3 = tf.get_variable("W3", [cand_num], trainable=True) #(cand_num)
            mean_eb = tf.reduce_mean(self.mid_embeddings_var, axis=0) #(embed_dim)
            candidates = tf.einsum('i,j->ij',self.W3, mean_eb) #(cand_num, embed_dim)
            self.candidates = tf.math.tanh(tf.einsum('ij,jk->ik',candidates, self.W1))#(cand_num, att_dim)
        else:
            self.candidates = tf.get_variable("candidates_pool", [cand_num, att_dim * 4], trainable=True) #(cand_num, att_dim)
        if f_trans:
            self.trans_matrix = tf.get_variable("trans_matrix", [cand_num, cand_num], trainable=True) #(cand_num, cand_num)
        #-------------------------------------------------------------------------------------------
        
        
        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = \
                tf.get_variable(
                    shape=[1, seq_len, embedding_dim],
                    name='position_embedding')
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb #(batch,seq_len,embed_dim)
            
        #-------------------------------------------------------------------------------------------
        with tf.variable_scope("activ_cand", reuse=tf.AUTO_REUSE) as scope:                        
            print('*'*25+'activ cand work'+'*'*25)
            masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1) #(batch,seq_len,embed_dim)
            item_his_eb_mean = tf.reduce_sum(item_list_add_pos, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9) #(batch,embed_dim)
            mean_hidden = tf.einsum('ij,jk->ik',item_his_eb_mean,self.W1) #(batch,att_dim)
            logits_out = tf.einsum('ij,kj->ik', mean_hidden, self.candidates) #(batch,cand_num)
            
            if f_trans:
                print('*'*25+'my trans work'+'*'*25)
                seq_eb_hidden = tf.einsum('ijk,kl ->ijl', item_list_add_pos, self.W1) #(batch, seq_len, att_dim)
                seq_cand_score = tf.einsum('ijl, kl->ijk', seq_eb_hidden, self.candidates) #(batch, seq_len, cand_num)
                seq_cand_hit = tf.math.argmax(seq_cand_score, axis=-1) #(batch, seq_len)
                seq_trans_score = tf.gather(self.trans_matrix, seq_cand_hit) #(batch,seq_len,cand_num)
                seq_trans_score = tf.reduce_mean(seq_trans_score, axis=1) #(batch, cand_num)
                logits_out = tf.multiply(logits_out, tf.nn.softmax(seq_trans_score))
            
            indices_K = tf.argsort(logits_out, axis=-1, direction='DESCENDING')[:,:num_interest] #(batch,intere_num)   
            activ_query = tf.gather(self.candidates, indices_K) #(batch,intere_num,att_dim)   
            
        #-------------------------------------------------------------------------------------------    
            
        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            item_hidden = tf.math.tanh(tf.einsum('ijk,kl->ijl',item_list_add_pos,self.W1)) # (batch,seq_len,att_dim)
            item_att_w =  tf.einsum('ijk,ilk->ijl', item_hidden, activ_query) # (batch,seq_len,intere_num)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1]) # (batch,intere_num,seq_len)

            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1]) #(batch,intere_num,seq_len)
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1) #(batch,intere_num,seq_len)

            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w) #(batch,intere_num,seq_len)
            item_att_w = tf.nn.softmax(item_att_w) #(batch,intere_num,seq_len)

            interest_emb = tf.matmul(item_att_w, item_list_emb) #(batch,intere_num,embed_dim)

        self.user_eb = interest_emb #(batch,intere_num,embed_dim)

        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1])) #(batch,intere_num,1)
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1)) #(batch,intere_num)

        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads) #(batch,embed_dim)

        self.build_sampled_softmax_loss(self.item_eb, readout)