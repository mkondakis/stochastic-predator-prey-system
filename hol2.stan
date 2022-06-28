functions { 
  real[] disc_equation(real t,
                       real[] z,
                       real[] theta,
                       real[]x_r,
                       int[]x_i) {
    real n = z[1];      
    real alpha = theta[1];
    real T = theta[2];
    real  pe = x_i[1];
    real dn_dt = (-alpha * n * pe) /( 1 + alpha * T * n)  ;
    return  {dn_dt} ;
  }
}
data {
  int N; // Number of time passes
  int M; //   M=1
  int NN; //number of data
  int n_trials;
  int<lower=0> pe[M]; // number of predators
  int<lower = 1> n_pars; // Number of model parameters
  int<lower=0> Ne[NN];// Number of consumed prey densities
  int<lower=0> A[NN]; // initial prey density
  real t[N];
  real t0;
}
transformed data {
  real x_r[0];
  //int x_i[0];

}
parameters{
  real <lower=0> theta[n_pars];// theta = { alpha,T}
}
transformed parameters{
  real<lower=0> N0[M]; // Initial population of prey densities
  real<lower=0> N_t[N,M];
  real<lower=0> p[NN];
 
      // pp [1]= pe;

  for (i in 1:NN){
     N0 = A[{i}];           // N1[i]=N0[i];
  N_t =  integrate_ode_rk45(disc_equation, N0, t0, t, theta, x_r, pe, 1e-4,1e-4,1e3);
    p[i] = (N0[M]-N_t[N,M])/N0[M];
  }
}
model{
  target+=4*beta_lpdf(theta[{1}]|1,1);
  target+=2*beta_lpdf(theta[{2}]|1,1);
  target+=binomial_lpmf(Ne|A,p);

}
generated quantities{
  real log_lik[NN];
  real y_fit[NN];
  real dev;
  
for (i in 1:NN) {
  log_lik[i]= binomial_lpmf(Ne[i] |A[i] , p[i]);
   y_fit[i]= binomial_rng(A[i] , p[i]);
}  
  dev = -2*sum(log_lik[]);
}

