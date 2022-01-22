import tensorflow as tf

class MultiDAE(tf.keras.Model):
    
    def __init__(self, p_dims, q_dims=None, dropout_prob=0, lam=0.01, lr=1e-3, random_seed=None):
        # MLP layer 차원 설정
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else: 
            assert q_dims[0] == p_dims[-1], "Input dims != output dims"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p != Latent dimension for q."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:] # 전체 dimension
        
        self.dropout_prob = dropout_prob
        self.lam = lam # for regularization
        self.lr = lr   # learning rate
        self.random_seed = random_seed
    
    def build(self, inputs_shape):
        self.construct_weights()
    
    def call(self, data, trainig=None):
        
        X = data
        h = tf.nn.l2_normalize(X, 1)
        h = tf.nn.dropout(h, self.dropout_prob)
        
        for i, (w, b) in enumerate(zip(self.weights_, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_) - 1:
                h = tf.nn.tanh(h)
        return  h
    
    def train_step(self, data):
        
        X = data
        return_loss = None
        
        with tf.GradientTape() as tape:
            pred = self(X, training=True)  # Forward pass
            return_loss = self.loss(X, pred)
            
        gradients = tape.gradient(return_loss, self.weights_)   
        self.optimizer.learning_rate = self.lr
        self.optimizer.apply_gradients(zip(gradients, self.weights_))
        
        return {'loss': return_loss}
    
    def loss(self, X, pred):
        logits = pred
        log_softmax_var = tf.nn.log_softmax(logits)
        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * X, axis=-1))
        regularizer = tf.keras.regularizers.L2(self.lam)
        reg_var = 0
        weights = self.weights_
        for i in range(len(weights)):
            reg_var += regularizer(weights[i])
        reg_var /= len(weights)
        neg_ELBO = neg_ll + 2 * reg_var
        return neg_ELBO
    
    def test(self, data):
        X = data
        logits = self(X, training=False)
        return logits

    # 가중치 초기화
    def construct_weights(self):
        
        self.weights_ = [] # 
        self.biases = []
        
        # define weights
        # d_in  : 1 2 3 4
        # d_out : 2 3 4 5
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            xaiver=tf.keras.initializers.glorot_uniform(seed=self.random_seed)
            initial_value = tf.Variable(xaiver(shape=[d_in, d_out]))
            self.weights_.append(tf.Variable(initial_value = initial_value,
                name=weight_key, shape=[d_in, d_out]))
            
            normal = tf.keras.initializers.TruncatedNormal(stddev=0.001, seed=self.random_seed)
            initial_value = tf.Variable(normal(shape=[d_out]))
            self.biases.append(tf.Variable(initial_value = initial_value, name=bias_key, shape=[d_out]))

class MultiVAE(tf.keras.Model):
    
    def __init__(self, p_dims, dropout_prob=0, q_dims=None, anneal=0, lam=0.01, lr=1e-3, random_seed=None):
        # MLP layer 차원 설정
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else: 
            assert q_dims[0] == p_dims[-1], "Input dims != output dims"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p != Latent dimension for q."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:] # 전체 dimension
        
        self.dropout_prob = dropout_prob
        self.lam = lam # for regularization
        self.lr = lr   # learning rate
        self.random_seed = random_seed
        self.anneal = 0
    
    def q_net(self, X):
        
        mu_q, std_q, KL = None, None, None
        h = tf.nn.l2_normalize(X, 1)
        h = tf.nn.dropout(h, self.dropout_prob)
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b     
            if i != len(self.weights_q) - 1: # 마지막 layer가 아닐 경우에는
                h = tf.nn.tanh(h)            # 활성화함수: tanh
            else: # 마지막 layer일 경우,
                mu_q = h[:, :self.q_dims[-1]] # 결과값 전까지 모든 값
                logvar_q = h[:, self.q_dims[-1]:]
                
                std_q = tf.exp(0.5 * logvar_q) # 결과값 전까지의 표준편차
                KL = tf.reduce_mean(tf.reduce_sum(
                        0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        
        return mu_q, std_q, KL
    
    def p_net(self, z):
        h = z
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b
            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h
    
    def build(self, inputs_shape):
        self.construct_weights()
    
    def call(self, data, trainig=None):
        
        X, is_training = data
        mu_q, std_q, KL = None, None, None
        
        mu_q, std_q, KL = self.q_net(X)
            
        epsilon = tf.random.normal(tf.shape(std_q))
        sampled_z = mu_q + is_training * epsilon * std_q

        logits = self.p_net(sampled_z)
        return logits, KL
    
    def train_step(self, data):
        X = data
        logits, KL, neg_ELBO, neg_ll = None, None, None, None
        
        with tf.GradientTape() as tape:
            pred = self((X, float(1)), training=True)  # Forward pass
            neg_ELBO = self.loss(X, pred)
            
        gradients = tape.gradient(neg_ELBO, self.weights_q + self.weights_p)   
        self.optimizer.learning_rate = self.lr
        self.optimizer.apply_gradients(zip(gradients, self.weights_q+ self.weights_p))
        
        return {'loss': neg_ELBO}
    
    def loss(self, X, pred):
        logits = pred[0]
        KL = pred[1]
        log_softmax_var = tf.nn.log_softmax(logits)
        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * X, axis=-1))
        regularizer = tf.keras.regularizers.L2(self.lam)
        reg_var = 0
        weights = self.weights_q + self.weights_p
        for i in range(len(weights)):
            reg_var += regularizer(weights[i])
        reg_var /= len(weights)
        neg_ELBO = neg_ll + self.anneal * KL + 2 * reg_var
        return neg_ELBO
    
    def test(self, data):
        X = data
        logits, KL = self((X, float(0)), training=False)
        return logits

    # 가중치 초기화
    def construct_weights(self):
        
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            
            xaiver=tf.keras.initializers.glorot_uniform(seed=self.random_seed)
            initial_value = tf.Variable(xaiver(shape=[d_in, d_out]))
            self.weights_q.append(tf.Variable(initial_value = initial_value,
                name=weight_key, shape=[d_in, d_out]))
            
            normal = tf.keras.initializers.TruncatedNormal(stddev=0.001, seed=self.random_seed)
            initial_value = tf.Variable(normal(shape=[d_out]))
            self.biases_q.append(tf.Variable(initial_value = initial_value, name=bias_key, shape=[d_out]))
            
        self.weights_p, self.biases_p = [], []
        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1) 
            
            xaiver=tf.keras.initializers.glorot_uniform(seed=self.random_seed)
            initial_value = tf.Variable(xaiver(shape=[d_in, d_out]))
            self.weights_p.append(tf.Variable(initial_value = initial_value,
                name=weight_key, shape=[d_in, d_out]))
            
            normal = tf.keras.initializers.TruncatedNormal(stddev=0.001, seed=self.random_seed)
            initial_value = tf.Variable(normal(shape=[d_out]))
            self.biases_p.append(tf.Variable(initial_value = initial_value, name=bias_key, shape=[d_out]))
        