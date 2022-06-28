functions { 
  real[] disc_equation(real t,
                       real[] z,
                       real[] theta,
                       real[]x_r,
                       int[]x_i) {
    real n = z[1];      
    real alpha = theta[1];
    real T = theta[2];
    real m = theta[3];
    int pe = x_i[1];
    
    real dn_dt = (-alpha * n * pow(pe,m)) /( 1 + alpha * T * n * pow(pe,m)) ;
    //real dn_dt = (-alpha * n * pe) /( 1 + alpha * T * n)  ;

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
  vector[3] ga=[85.55,126.40,141.65]';
  vector[3] gb=[298.77,380.66,763.21]';
}
parameters{
  real <lower=0, upper=1> theta[n_pars];// theta = { alpha,T}
}
transformed parameters{
//  real<lower=0> N0[M]; // Initial population of prey densities
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
  
  target+=binomial_lpmf(Ne|A,p);
 // target+=exponential_lpdf(theta[{1}]|5);
  target+=exponential_lpdf(theta[{2}]|1e-3);
  target+=exponential_lpdf(theta[{3}]|1e-3);
  target+=gamma_lpdf(theta[{1}]|ga[pe[1]-1],gb[pe[1]-1]);


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



