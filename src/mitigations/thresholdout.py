import numpy as np

class Thresholdout:
    def __init__(self, num_holdout_samples: int, threshold_factor: float = 0.3, noise_rate: float = 0.2, mean_label_to_scale=None):
        """
        Thresholdout implementation from Dwork et al.

        Note that there are some differences in the Thresholdout implementation from the paper's pseudocode and the
        code the authors provide with their publication. The main differences:
        - Pseudocode uses Laplacian noise, implementation uses Gaussian noise:
        -- 'Second, we used Gaussian noise instead of Laplacian noise as it has stronger concentration properties
            (in many differential privacy applications similar theoretical guarantees hold for mechanisms based
            on Gaussian noise—although not for ours).'

        - Pseudocode uses a Budget, that when exhausted does not give any holdout information
        - Threshold in pseudocode is an initial threshold (hyperparameter) + Lap(2*sigma), in implementation the threshold
          is set as hyperparameter

        - Pseudocode: Tolerance += Lap(4*noise_rate)
        - Implementation: Tolerance += Gauss(noise_rate)

        - Pseudocode: Threshold initialized as T + Lap(2*sigma) (so the first eval is T + Lap(2*noise_rate) + Lap(2 * noise_rate))
        - Implementation: Threshold initialized to T

        Args:
            num_holdout_samples (int): Number of samples in the holdout set.
            threshold_factor (float): Scaling factor for the threshold calculation. Default is 4.0.
            noise_rate (float): Scaling factor for the noise calculation. Default is 1.0.
            mean_label_to_scale (float, optional): If provided, the threshold and noise_rate will be scaled by this value (for regression)
        """

        # Standard parameterization from paper
        self.threshold = threshold_factor / np.sqrt(num_holdout_samples)  # T = 4/√n
        self.noise_rate = noise_rate / np.sqrt(num_holdout_samples)    # σ = 1/√n

        # If problem is regression, adding 0-1 scale noise does not make sense, 
        # In that case we scale with the mean label value
        # Thresholdout does not specify a case for this
        if mean_label_to_scale is not None:
            self.threshold *= mean_label_to_scale
            self.noise_rate *= mean_label_to_scale

    def query(self, train_score: float, holdout_score: float) -> float:
        """
        Determines whether to return the training accuracy or a modified holdout accuracy
        based on the similarity between the training and holdout scores. If the absolute
        difference between the given training and holdout scores is within a predefined
        threshold, the training score is returned. Otherwise, a noisy version of the
        holdout score is calculated and returned.

        Args:
            train_score: The accuracy score of the training dataset.
            holdout_score: The accuracy score of the holdout dataset.

        Returns:
            float: Either the training score or a noisy version of the holdout score,
            depending on the comparison between the training and holdout scores.
        """

        # Query logic for Thresholdout
        threshold = self.threshold + np.random.normal(0, self.noise_rate)
        # print('-------------------------------------')
        # print("Treshold:", threshold)
        # print("Diff:", abs(train_score - holdout_score))

        if abs(train_score - holdout_score) <= threshold:
            # Return training accuracy (no holdout information leaked)
            # print("Train", train_score, "Holdout", holdout_score, "=> Returning train score")
            return train_score

        else:
            # Return noisy holdout accuracy
            res = holdout_score + np.random.normal(0, self.noise_rate)
            # print("Train", train_score, "Holdout", holdout_score, "=> Returning noisy holdout score", res)

            return res
