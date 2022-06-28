functions { 
  real[] disc_equation(real t,
                       real[] z,
                       real[] theta,
                       real[]x_r,
                       int[]x_i) {
    real n = z[1];      
    real a = theta[1];
    real b = theta[2];
    real c = theta[3];
    int pe = x_i[1];
      // real dn_dt = (-a * n * pe) /( 1 + a * b * n + c * (pe-1)) ; //testing function response
    real dn_dt = pe * (sqrt((1 + a * n *((b+1)^2+(c+1)^2-2*(b*c+1))))- 1 - a * n * b-a * n * c ) /( 2* a * n * b * c) ;
    return  {dn_dt} ;
  }
}
data {
  int N; // number of time counts ex. number of hours prey exposed to predator
  int M; // dimension of ODE output
  int NN; // sample size
  int n_trials; // sample size
  int<lower=0> pe[M]; // number of predators
  int<lower = 1> n_pars; // Number of model parameters
  int<lower=0> Ne[NN]; // densities of consumed prey
  int<lower=0> A[NN]; // initial prey density
  real t[N]; // time vector at each count
  real t0; // initial time
  real<lower=0> gshape;
  real<lower=0> gscale;
}
transformed data { 
  real x_r[0];
  //int x_i[0];
  vector[3] ga=[gshape,126.40,141.65]';
  vector[3] gb=[gscale,380.66,763.21]';}
parameters{
  real <lower=0, upper=1> theta[n_pars];// theta = { alpha,T}
}
transformed parameters{

  real<lower=0> N0[M]; // Initial population of prey densities
  real<lower=0> N_t[N,M];
  real<lower=0, upper=1> p[NN];
 

  for (i in 1:NN){
     N0 = A[{i}];         
  //N_t =  integrate_ode_rk45(disc_equation, N0, t0, t, theta, x_r, pe, 1e-4,1e-4,1e3);
  N_t =  integrate_ode_rk45(disc_equation, N0, t0, t, theta, x_r, pe);
    p[i] = (N0[M]-N_t[N,M])/N0[M];
  }
}
model{
  
  target+=binomial_lpmf(Ne|A,p);
  target+=gamma_lpdf(theta[{2}]|ga[pe[1]],gb[pe[1]]);
  target+=exponential_lpdf(theta[{3}]|1e-1);
  target+=exponential_lpdf(theta[{1}]|1e-1);


  //target+=lognormal_lpdf(theta[{3}]|0,1);
  //target+=lognormal_lpdf(N0|10,1);
  //N_t[,1] ~ gamma( rep_row_vector(1, N), rep_row_vector(1, N));
  
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



