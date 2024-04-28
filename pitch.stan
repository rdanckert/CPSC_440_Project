
data {
  int<lower=0> N;
  int<lower=0> N_valid;
  vector[N] xwoba;
  vector[N] ff_percent; //fastball
  vector[N] ff_avg_velo;
  vector[N] ff_avg_spin;
  vector[N] ff_avg_h_break; //horizontal break
  vector[N] ff_avg_v_break; //vertical break
  vector[N] sl_percent; //slider
  vector[N] sl_avg_velo;
  vector[N] sl_avg_spin;
  vector[N] sl_avg_h_break;
  vector[N] sl_avg_v_break;
  vector[N] ch_percent; //changeup
  vector[N] ch_avg_velo;
  vector[N] ch_avg_spin;
  vector[N] ch_avg_h_break;
  vector[N] ch_avg_v_break;
  vector[N_valid] ff_percent_pred; //fastball
  vector[N_valid] ff_avg_velo_pred;
  vector[N_valid] ff_avg_spin_pred;
  vector[N_valid] ff_avg_h_break_pred; //horizontal break
  vector[N_valid] ff_avg_v_break_pred; //vertical break
  vector[N_valid] sl_percent_pred; //slider
  vector[N_valid] sl_avg_velo_pred;
  vector[N_valid] sl_avg_spin_pred;
  vector[N_valid] sl_avg_h_break_pred;
  vector[N_valid] sl_avg_v_break_pred;
  vector[N_valid] ch_percent_pred; //changeup
  vector[N_valid] ch_avg_velo_pred;
  vector[N_valid] ch_avg_spin_pred;
  vector[N_valid] ch_avg_h_break_pred;
  vector[N_valid] ch_avg_v_break_pred;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real ff_percent_slope; //fastball
  real ff_avg_velo_slope;
  real ff_avg_spin_slope;
  real ff_avg_h_break_slope; //horizontal break
  real ff_avg_v_break_slope; //vertical break
  real sl_percent_slope; //slider
  real sl_avg_velo_slope;
  real sl_avg_spin_slope;
  real sl_avg_h_break_slope;
  real sl_avg_v_break_slope;
  real ch_percent_slope; //changeup
  real ch_avg_velo_slope;
  real ch_avg_spin_slope;
  real ch_avg_h_break_slope;
  real ch_avg_v_break_slope;
  real<lower=0> sigma;
  real<lower=0> param_sigma;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // priors
  param_sigma ~ exponential(1);
  sigma ~ exponential(1);
  ff_percent_slope ~ normal(0,param_sigma); //fastball
  ff_avg_velo_slope ~ normal(0,param_sigma);
  ff_avg_spin_slope ~ normal(0,param_sigma);
  ff_avg_h_break_slope ~ normal(0,param_sigma); //horizontal break
  ff_avg_v_break_slope ~ normal(0,param_sigma); //vertical break
  sl_percent_slope ~ normal(0,param_sigma); //slider
  sl_avg_velo_slope ~ normal(0,param_sigma);
  sl_avg_spin_slope ~ normal(0,param_sigma);
  sl_avg_h_break_slope ~ normal(0,param_sigma);
  sl_avg_v_break_slope ~ normal(0,param_sigma);
  ch_percent_slope ~ normal(0,param_sigma); //changeup
  ch_avg_velo_slope ~ normal(0,param_sigma);
  ch_avg_spin_slope ~ normal(0,param_sigma);
  ch_avg_h_break_slope ~ normal(0,param_sigma);
  ch_avg_v_break_slope ~ normal(0,param_sigma);
  
  
  // likelihood
  xwoba ~ normal(ff_percent_slope*ff_percent + ff_avg_velo_slope*ff_avg_velo 
  + ff_avg_spin_slope*ff_avg_spin + ff_avg_h_break_slope*ff_avg_h_break + 
  ff_avg_v_break_slope*ff_avg_v_break + sl_percent_slope*sl_percent + 
  sl_avg_velo_slope*sl_avg_velo + sl_avg_spin_slope*sl_avg_spin + 
  sl_avg_h_break_slope*sl_avg_h_break + sl_avg_v_break_slope*sl_avg_v_break + 
  ch_percent_slope*ch_percent + ch_avg_velo_slope*ch_avg_velo + 
  ch_avg_spin_slope*ch_avg_spin + ch_avg_h_break_slope*ch_avg_h_break + 
  ch_avg_v_break_slope*ch_avg_v_break, sigma);
}

generated quantities {
  vector[N_valid] xwoba_pred;
  for (i in 1:N_valid)
    xwoba_pred[i] = normal_rng(ff_percent_pred[i]*ff_percent_slope + 
    ff_avg_velo_slope*ff_avg_velo_pred[i] + ff_avg_spin_slope*ff_avg_spin_pred[i]
    + ff_avg_h_break_slope*ff_avg_h_break_pred[i] + ff_avg_v_break_slope*ff_avg_v_break_pred[i]
    + sl_percent_slope*sl_percent_pred[i] + sl_avg_velo_slope*sl_avg_velo_pred[i] 
    + sl_avg_spin_slope*sl_avg_spin_pred[i] + sl_avg_h_break_slope*sl_avg_h_break_pred[i]
    + sl_avg_v_break_slope*sl_avg_v_break_pred[i] + ch_percent_slope*ch_percent_pred[i] 
    + ch_avg_velo_slope*ch_avg_velo_pred[i] + ch_avg_spin_slope*ch_avg_spin_pred[i] 
    + ch_avg_h_break_slope*ch_avg_h_break_pred[i] + 
    ch_avg_v_break_slope*ch_avg_v_break_pred[i], sigma);
    
}
