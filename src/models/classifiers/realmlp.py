"""
To integrate RealMLP, you need the search space, which is on page 45 of the RealMLP paper:
https://arxiv.org/pdf/2407.04491

With that search space, it seems you can use the standard RealMPL-TD (tuned defaults) model, where you pass the
extra hyperparameters to the model (the rest is a tuned default).

Q: Can we use the TD? Are the defaults not tuned on some of the TabArena datasets? What does TabArena do?

Several search spaces (but let's use the one from the paper?)
https://github.com/dholzmueller/pytabkit/blob/main/pytabkit/models/alg_interfaces/nn_interfaces.py#L368

An example of manual HPO:
https://pytabkit.readthedocs.io/en/latest/models/02_hpo.html

Q: How to do the CV ourselves? Should/can we do that? We do not have gpu to our disposal of course

From paper:
For best performance, it might be beneficial to use a larger search space for
the init standard deviation of the first embedding layer, and to tune the embedding dimensions, as in
Table C.16. (see table c16 below)
Hyperparameter Space
MLP hyperparameters as in Table C.15
Num. emb. type PLR
Num. emb. initialization σ LogUniform[1e-2, 1e1]
Num. emb. #frequencies Uniform[1, 64]
Num. emb. dimension Uniform[1, 64]

Q: Do we want to do anything with the probabilities in the search space? I guess they are defined for RS, but we do BO.
Q: What about validation data in the RealMLP? Is used for early stopping. But do we use our HPO validation data?
If not, do we retrain before validation?
"""

from pytabkit import RealMLP_TD_Classifier
from src.models.base_component import BaseComponent


class RealMLPModel(BaseComponent):
    def __init__(self):
        # Initialize model with prefix
        super().__init__("RealMLP")

        # Required hyperparameters
        self.required_hyperparameters = [
            f"{self.prefix}_num_embedding_type",
            f"{self.prefix}_use_scaling_layer",
            f"{self.prefix}_learning_rate",
            f"{self.prefix}_dropout_prob",
            f"{self.prefix}_activation_function",
            f"{self.prefix}_hidden_layer_sizes",
            f"{self.prefix}_weight_decay",
            f"{self.prefix}_w_init_std",
        ]

    def construct(self, config):
        """Construct MLPClassifier with hyperparameters
        """
        model_threads = max(1, int(config.get("_model_threads", 1)))

        # Store hyperparameters relevant for MLPClassifier in configs
        self.config = {
            k: v for k, v in config.items() if k.startswith(self.prefix) or k == "random_state"
        }

        # Check hyperparameters
        self.check(self.config)

        hidden_size_choices = [[256] * 3, [512], [64] * 5]

        return RealMLP_TD_Classifier(
            hidden_sizes=hidden_size_choices[self.config[f"{self.prefix}_hidden_layer_sizes"]], # Note that hidden size choices should be an integer 0, 1, 2
            num_emb_type=self.config[f"{self.prefix}_num_embedding_type"], # Should be none, PBLD, PL, PLR
            act=self.config[f"{self.prefix}_activation_function"], # Should be ReLU, SELU, Mish
            p_drop=self.config[f"{self.prefix}_dropout_prob"], # Choice of 0, 0.15, 0.3
            wd=self.config[f"{self.prefix}_weight_decay"], # Should be choice of 0 and 2e-2
            lr=self.config[f"{self.prefix}_learning_rate"], # LogUniform([2e-2, 3e-1])
            add_front_scale=self.config[f"{self.prefix}_use_scaling_layer"], # True or False
            use_ls=False,
            ls_eps=0.0,
            plr_sigma=self.config[f"{self.prefix}_w_init_std"], # LogUniform([0.05, 0.5])
            device="cpu",
            random_state=self.config["random_state"],
            val_metric_name="1-auc_ovr",
            verbosity=0,
            n_threads=model_threads,
            n_cv=1,
            n_refit=0,
            n_ens=1,
            # use_early_stopping=True,
            # early_stopping_additive_patience=16,
            # early_stopping_multiplicative_patience=1,
        )
