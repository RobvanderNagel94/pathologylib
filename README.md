## Features
This library presents key EEG domain features, among others:

-	Background features (i.e. alpha rhythm frequency, reactivity, anterio–posterior gradient, interhemispheric asymmetries, diffuse slow-wave activity)
- Time-frequency features (i.e. STFT, welch, mNNC, COGX, COGY)
-	Statistical features (i.e. skewness, kurtosis, line length, maximum, minimum, mean, median, lumpiness, flat spots, zero crossing, zero crossing derivative)
- Time complexity features (i.e. Hjorth activity, mobility, and complexity, (nonlinear) energy, Higuchi and Petrosian fractal dimension, largest Lyapunov exponent, Hurst exponent, svd entropy and Fisher information)
- Connectivity features (i.e. phase locking values)
- Frequency features (i.e. discrete and continuous wavelet transformation, Fourier transformation)

## File Structure
The library consists of the following files:
- core: Contains the core code that defines fundamental components for EEG data analysis, such as EEG dataset handling and montage configurations.
- datasets: Contains code for managing EEG datasets, including loading, preprocessing, and splitting into training and testing sets.
- features: Contains code for extracting features from EEG data. This includes both multi- and single-channel signal features that are essential for training machine learning models.
- models: Holds the machine learning models implemented for EEG classification tasks. These models are designed to process the extracted features and make predictions.
- research: Contains research papers, articles, and references related to the EEG classification project. This directory helps maintain a collection of relevant literature.
- utils: Contains utility functions and helper code that is shared across different parts of the project. These utilities assist in tasks like data manipulation and processing.
- validation: Incorporates code for validating models, offering insights into experiment reliability and performance assessment.
- visualization: Contains code for visualizing the results and outputs of the trained models. This helps in gaining insights into the model's performance and the patterns it has learned.

## Usage
The library is mainly to provide some basic functionality that can be integrated into any EEG-based research project. Import the necessary functions from the relevant files, extract the features from the EEG signals, and train machine-learning models using the provided code!

## Dependencies

- Python >= 3.7
- NumPy >= 1.18.1
- SciPy >= 1.4.1
- Matplotlib >= 3.1.0
- Scikit-learn >= 0.22.0
- Pandas >= 1.0.0
