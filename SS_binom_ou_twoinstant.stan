functions { 
  real[] sss_equation(real t,
                       real[] z,
                       real[] theta,
                       real[]x_r,
                       int[]x_i) {
    real n = z[1];      
    real a = theta[1];
    real b = theta[2];
    real c = theta[3];
    int pe = x_i[1];
    
    real dn_dt = pe * (sqrt((1 + a * n *((b+1)^2+(c+1)^2-2*(b*c+1))))- 1 - a * n * b-a * n * c ) /( 2* a * n * b * c) ;
    
    //real dn_dt = (-alpha * n ) /( pow(pe,m)+ alpha * T * n ) ;

    return  {dn_dt} ;
  }
}
data {
  int N; // number of time counts ex. number of hours prey exposed to predator
  int M; // dimension of ODE output
  int NN; //sample size
  int n_trials; //sample size
  int<lower=0> pe[M]; // number of predators
  int<lower = 1> n_pars; // number of model parameters
  int<lower=0> Ne[NN];// densities of consumed prey
  int<lower=0> A[NN]; // initial prey density
  real t[N]; //time vector at each count
  real t0; //initial time
  real<lower=0> gshape;
  real<lower=0> gscale;
}
transformed data {
  real x_r[0];
  //int x_i[0];
  //real ga=gshape;
  //real gb=gscale;
  vector[3] ga=[85.55,126.40,141.65]';
  vector[3] gb=[298.77,380.66,763.21]';
 //vector[3] ga=[gshape,126.40,141.65]';
  //vector[3] gb=[gscale,380.66,763.21]';
}
parameters{
  real <lower=0, upper=1> theta[n_pars];// theta = { alpha,T}
  real<lower=0> phi;              // speed of reversion
  real<lower=0> s_sq;             // square of instantaneous diffusion term
  real kappa[2,NN];              // logarithm of poisson parameter
  //real sigma;                  // variance of OU process

}
transformed parameters{
  real<lower=0> N0[M]; // Initial population of prey densities
  real<lower=0> N_t[N,M];
  real<lower=0> pr[2,NN];
  real lambda[2,NN];          // poisson parameter
  real <lower=0> sigma;                  // variance of OU process
  for (i in 1:NN){
    N0 = A[{i}];           // N1[i]=N0[i];
    N_t =  integrate_ode_rk45(sss_equation, N0, t0, t, theta, x_r, pe,1e-4,1e-4,1e3);
    //N_t =  integrate_ode_adams(disc_equation, N0, t0, t, theta, x_r, pe,1e0,1e0,1e6);
    //N_t =  integrate_ode_bdf(disc_equation, N0, t0, t, theta, x_r, pe,1e-4,1e-4,1e5);
    
    pr[1,i] = (N0[M] -N_t[1,M])/N0[M];
    //for(j in 1:N-1) {
    //pr[2,i] = (N_t[j,M]-N_t[j-1,M]);
    //}
    pr[2,i] = (N0[M]-N_t[N,M])/N0[M];
    
  }
  lambda=inv_logit(kappa);
  sigma=(1-exp(-2*phi))*(s_sq/(2*phi));

}
model{
  real mu[N,NN];        //logarithm of solution of deterministic model
  //priors
  target+=gamma_lpdf(phi|0.1,0.1);
  target+=gamma_lpdf(s_sq|0.1,0.1); 
  target+=exponential_lpdf(theta[{1}]|1e-3);  
  target+=exponential_lpdf(theta[{2}]|1e-3);  
  //target+=gamma_lpdf(theta[{2}]|ga[pe[1]],gb[pe[1]]);
  target+=exponential_lpdf(theta[{3}]|1e-3);


//likelihood
  for (j in 1:NN){
  mu[1,j] =  logit(pr[1,j]);
  target+=normal_lpdf(kappa[1,j]|mu[1,j],sigma); // kappa=log(lambda) is an Ornstein-Uhlenbeck process
  mu[2,j] =  logit(pr[2,j]);
  target+=normal_lpdf(kappa[2,j]|mu[2,j]+(kappa[1,j]-mu[2,j])*exp(-phi),sigma);

  }
  target+=binomial_lpmf(Ne|A,lambda[2,]); // lambda has depends on the observational times
 
}
generated quantities{
  real log_lik[NN];
  real y_fit[NN];
  real dev;
for (i in 1:NN) {
  log_lik[i] = binomial_lpmf(Ne[i]|A[i],lambda[2,i]);
  y_fit[i]= binomial_rng(A[i],lambda[2,i]);
} 
  dev = -2*sum(log_lik[]);
}
