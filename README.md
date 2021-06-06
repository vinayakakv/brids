# BRIDS
Border Router Intrusion Detection System. _It's not birds, btw!_

# What?
An intrusion detection system using network-flow statistics collected at a border router. A minor project done for Advanced Computer Networks course at Department of IT, NITK.

# Goal
To access the performance of the 1D-CNN, and Random Forest classifier on unormalized and normalized network-flow statistics, with and without autoencoder.

# Methods and Techniques
- Used `dvc` (https://dvc.org/) for pipeline construction)
- Tensorflow 2 with `tf.keras` for Neural network
- `scikit-learn` for Random Forest

# Results
- Random Forest works well both with normalized and unnormalized data
- Neural networks needs normalized data for better performance
- Auto-encoders do a better job in encoding unnormalized data, as evident by the performace of `Unnormalized+Auto-encoder+Random Forest` configuration

# Summary of the work
The methods and results were supposed to be published at a conference, but was rejected as the paper was poorly written and conclusions were not significant. It was communicated to another conference as-is, which got accepted with a suggestion to make the title short 🤣; which was then dropped by me. The pdf of the paper is [provided](ids_paper.pdf), just for reference. **It is not to be taken seriously**.
