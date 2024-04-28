suppressPackageStartupMessages(require(tidyverse))

pitch_train <- read_csv("C:/Users/ranen/OneDrive - UBC/Desktop/CPSC 440/pitch_train")
pitch_valid <- read_csv("C:/Users/ranen/OneDrive - UBC/Desktop/CPSC 440/pitch_valid")
xwoba_train <- read_csv("C:/Users/ranen/OneDrive - UBC/Desktop/CPSC 440/xwoba_train")
xwoba_valid <- read_csv("C:/Users/ranen/OneDrive - UBC/Desktop/CPSC 440/xwoba_valid")

xwoba_train_vec <- xwoba_train$xwoba
xwoba_valid_vec <- xwoba_valid$xwoba

suppressPackageStartupMessages(require(rstan))

fit = stan(
  "pitch.stan", 
  seed = 1,
  data = list(
    xwoba = xwoba_train_vec,
    ff_percent = pitch_train$n_ff_formatted,
    ff_avg_velo = pitch_train$ff_avg_speed,
    ff_avg_spin = pitch_train$ff_avg_spin,
    ff_avg_h_break = pitch_train$ff_avg_break_x,
    ff_avg_v_break = pitch_train$ff_avg_break_z,
    sl_percent = pitch_train$n_sl_formatted,
    sl_avg_velo = pitch_train$sl_avg_speed,
    sl_avg_spin = pitch_train$sl_avg_spin,
    sl_avg_h_break = pitch_train$sl_avg_break_x,
    sl_avg_v_break = pitch_train$sl_avg_break_z,
    ch_percent = pitch_train$n_ch_formatted,
    ch_avg_velo = pitch_train$ch_avg_speed,
    ch_avg_spin = pitch_train$ch_avg_spin,
    ch_avg_h_break = pitch_train$ch_avg_break_x,
    ch_avg_v_break = pitch_train$ch_avg_break_z,
    N = length(xwoba_train_vec)
  ), 
  chains = 2,
  iter = 10000      
)

suppressPackageStartupMessages(require(bayesplot))

# Trace plots
mcmc_trace(fit, pars = c("ff_percent_slope")) + theme_minimal() + ylab("fastball %")
mcmc_trace(fit, pars = c("ff_avg_velo_slope")) + theme_minimal() + ylab("fastball velocity")
mcmc_trace(fit, pars = c("ff_avg_spin_slope")) + theme_minimal() + ylab("fastball spin")
mcmc_trace(fit, pars = c("ff_avg_h_break_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("ff_avg_v_break_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("sl_percent_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("sl_avg_velo_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("sl_avg_spin_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("sl_avg_h_break_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("sl_avg_v_break_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("ch_percent_slope","ch_avg_velo_slope",
                         "ch_avg_spin_slope", "ch_avg_h_break_slope",
                         "ch_avg_v_break_slope")) + theme_minimal()
mcmc_trace(fit, pars = c("param_sigma")) + theme_minimal() + ylab("parameter standard deviation")
mcmc_trace(fit, pars = c("sigma")) + theme_minimal() + ylab("linear model standard deviation")


mcmc_rank_hist(fit, pars = c("ff_avg_velo_slope")) + theme_minimal() + ylab("fastball velocity")
mcmc_rank_hist(fit, pars = c("ff_avg_spin_slope")) + theme_minimal() + ylab("fastball spin")

calibration_fit = stan(
  "pitch.stan", 
  seed = 1,
  data = list(
    xwoba = xwoba_train_vec,
    ff_percent = pitch_train$n_ff_formatted,
    ff_avg_velo = pitch_train$ff_avg_speed,
    ff_avg_spin = pitch_train$ff_avg_spin,
    ff_avg_h_break = pitch_train$ff_avg_break_x,
    ff_avg_v_break = pitch_train$ff_avg_break_z,
    sl_percent = pitch_train$n_sl_formatted,
    sl_avg_velo = pitch_train$sl_avg_speed,
    sl_avg_spin = pitch_train$sl_avg_spin,
    sl_avg_h_break = pitch_train$sl_avg_break_x,
    sl_avg_v_break = pitch_train$sl_avg_break_z,
    ch_percent = pitch_train$n_ch_formatted,
    ch_avg_velo = pitch_train$ch_avg_speed,
    ch_avg_spin = pitch_train$ch_avg_spin,
    ch_avg_h_break = pitch_train$ch_avg_break_x,
    ch_avg_v_break = pitch_train$ch_avg_break_z,
    N = length(xwoba_train_vec),
    N_valid = length(xwoba_valid_vec),
    ff_percent_pred = pitch_valid$n_ff_formatted,
    ff_avg_velo_pred = pitch_valid$ff_avg_speed,
    ff_avg_spin_pred = pitch_valid$ff_avg_spin,
    ff_avg_h_break_pred = pitch_valid$ff_avg_break_x,
    ff_avg_v_break_pred = pitch_valid$ff_avg_break_z,
    sl_percent_pred = pitch_valid$n_sl_formatted,
    sl_avg_velo_pred = pitch_valid$sl_avg_speed,
    sl_avg_spin_pred = pitch_valid$sl_avg_spin,
    sl_avg_h_break_pred = pitch_valid$sl_avg_break_x,
    sl_avg_v_break_pred = pitch_valid$sl_avg_break_z,
    ch_percent_pred = pitch_valid$n_ch_formatted,
    ch_avg_velo_pred = pitch_valid$ch_avg_speed,
    ch_avg_spin_pred = pitch_valid$ch_avg_spin,
    ch_avg_h_break_pred = pitch_valid$ch_avg_break_x,
    ch_avg_v_break_pred = pitch_valid$ch_avg_break_z
  ), 
  chains = 2,
  iter = 10000      
)

calibration_quantiles <- summary(calibration_fit, probs = c(0.025, 0.975))$summary
calibration_quantiles <- calibration_quantiles[,c("2.5%", "97.5%")]
calibration_quantiles <- 
  calibration_quantiles[!rownames(calibration_quantiles) %in% 
                          c("ff_percent_slope", "ff_avg_velo_slope", "ff_avg_spin_slope",
                            "ff_avg_h_break_slope", "ff_avg_v_break_slope", "sl_percent_slope",
                            "sl_avg_velo_slope", "sl_avg_spin_slope",
                            "sl_avg_h_break_slope", "sl_avg_v_break_slope", 
                            "ch_percent_slope", "ch_avg_velo_slope", "ch_avg_spin_slope",
                            "ch_avg_h_break_slope", "ch_avg_v_break_slope", "lp__", 
                            "sigma", "param_sigma"),]

inside <- 0
n_valid <- nrow(xwoba_valid)
for  (i in 1:n_valid) {
  if(xwoba_valid$xwoba[i] > calibration_quantiles[i,1] && 
     xwoba_valid$xwoba[i] < calibration_quantiles[i,2]) {
    inside <- inside + 1 
  }
}
inside/n_valid

calibration_tib <- 
  tibble("xwOBA" = xwoba_valid_vec, 
         "lower bound" = calibration_quantiles[,1], 
         "upper bound" = calibration_quantiles[,2])

ggplot() +
  geom_point(aes(x = seq(1,25), y=xwoba_valid_vec[1:25]), size = 4) +
  geom_point(aes(x = seq(1,25), y = calibration_tib$`lower bound`[1:25]), colour = "red", size = 4) +
  geom_point(aes(x = seq(1,25), y = calibration_tib$`upper bound`[1:25]), colour = "blue", size = 4) +
  labs(y = "xwOBA",x = "") +
  theme_minimal()

slope_estimates_summary <- summary(calibration_fit, probs = c(0.025, 0.5, 0.975))$summary
slope_estimates_summary <- slope_estimates_summary[,c("2.5%","50%", "97.5%")]
slope_estimates_summary <- 
  slope_estimates_summary[rownames(slope_estimates_summary) %in% 
                          c("ff_percent_slope", "ff_avg_velo_slope", "ff_avg_spin_slope",
                            "ff_avg_h_break_slope", "ff_avg_v_break_slope", "sl_percent_slope",
                            "sl_avg_velo_slope", "sl_avg_spin_slope",
                            "sl_avg_h_break_slope", "sl_avg_v_break_slope", 
                            "ch_percent_slope", "ch_avg_velo_slope", "ch_avg_spin_slope",
                            "ch_avg_h_break_slope", "ch_avg_v_break_slope"),]

write_csv(data.frame(slope_estimates_summary), "mcmc_estimates")
