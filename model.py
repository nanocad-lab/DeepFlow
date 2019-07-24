class Model:
  def __init__(self, exp_config):
      self.batch_size       = exp_config.model_config.batch_size
      self.vocab_size       = exp_config.model_config.vocab_size
      self.num_layers       = exp_config.model_config.num_layers
      self.hidden_dim       = exp_config.model_config.layer_size
      self.projection       = exp_config.model_config.projection
      self.seq_len          = exp_config.model_config.seq_len
      self.num_gates        = exp_config.model_config.num_gates
      self.num_non_linear   = exp_config.model_config.num_non_linear
      self.num_add          = exp_config.model_config.num_add
      self.num_pointwise    = self.num_non_linear + self.num_add
        

