## Features
This library presents key EEG domain features, among others:

-	Background features (i.e., alpha rhythm frequency, reactivity, anterio–posterior gradient, interhemispheric asymmetries, diffuse slow-wave activity)
- Spatial features (i.e., stft, welch, mnnc, cogx, cogy)
-	Statistical features (i.e., skewness, kurtosis, line length, maximum, minimum, mean, median, lumpiness, flat spots, zero crossing, zero crossing derivative)
- Time complexity features (i.e., hjorth activity, mobility, and complexity, (nonlinear) energy, higuchi and petrosian fractal dimension, largest lyapunov exponent, hurst exponent, svd entropy and fisher information)
- Connectivity features (i.e., phase locking values)
- Frequency features (i.e., discrete and continious wavelet transformation, fourier transformation)

## File Structure
The library consists of the following files:
- core: Contains the core code to define an EEG dataset and montages.
-	eval: Contains the evaluation code for the trained models.
-	features: Contains code for extracting both multi and single channel signal features.
-	models: Contains the machine learning models used for classification.
-	research: Contains research papers and articles related to the project.
-	visualization: Contains code for visualizing the results of the models.

## Usage
The library is mainly to provide some basic functionality that can be integrated into any EEG-based research project. Simply import the necessary functions from the relevant files, extract the features from the EEG signals, and train machine learning models using the provided code!

## Dependencies

- Python >= 3.7
- NumPy >= 1.18.1
- SciPy >= 1.4.1
- Matplotlib >= 3.1.0
- Scikit-learn >= 0.22.0
- Pandas >= 1.0.0
