# Reinforcement Learning - Learned Indicators
---

Read Me to be updated

## Filler Guide
---
### Dataset Downloading:
Dataset can be downloaded using the following repo:
https://github.com/Campbell-Rankine/YahooFinanceScraper

To download from the repository run "python data.py --ind True"

### Running the model:
Development has been focused over a latent version of the total structure, therefore at this point 'base' and 'optimize' may have bugs or may not work at all.
To run call:
python Main.py --mode latent (if you would like to test over 1 epoch add the additional flag: --debug True)

### Tensorboard:
Can view testing metrics on a local tensorboard server. Built in metrics include:
Net Worth
Stock Price
Profit

Plans to add the following metrics are underway:
PSNR - AutoEncoder Reconstruction metric (may not be entirely useful as we never need to reconstruct the data however gives us some kind of measure for the noise added by using the AE)
Actions at each timestep (Including logging the probability of taking each action)
Built in support will be provided so that the user can pass their own custom functions to the testing function and they will be displayed in tensorboard

### To Do:
1) Implement testing suite.
2) Make trader class more abstract
3) Attempt to move to a more centralized approach