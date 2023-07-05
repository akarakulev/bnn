# Playground to test Bayesian Neural Networks

- `bnn.py`
    - Posterior is a fully factorized gaussian
    - Prior is a fully factorized gaussian (KL-divergence has a simple closed form)
    - ELBO is approximated with Local Reparametrization Trick => more efficient
    - `Classification.ipynb` – MNIST classification
    - Reference (for Local Reparametrization): [Kingma et al., 2015](https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf) 

- `Blundell/blundell.py`
    - Posterior is a fully factorized gaussian
    - Prior is a mixture of two fully factorized gaussians
    - ELBO is resampled several times to get one stochastic approximation => not too efficient
    
    - `Blundell/Classification Blundell.ipynb` – MNIST classification
    - `Blundell/Regression Blundell.ipynb` – 1-d curve fitting
    - Reference: [Blundell et al., 2015](https://arxiv.org/pdf/1505.05424.pdf)