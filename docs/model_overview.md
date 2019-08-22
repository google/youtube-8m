# Overview of Models

This sample code contains implementations of the models given in the
[YouTube-8M technical report](https://arxiv.org/abs/1609.08675).

## Video-Level Models

*   `LogisticModel`: Linear projection of the output features into the label
    space, followed by a sigmoid function to convert logit values to
    probabilities.
*   `MoeModel`: A per-class softmax distribution over a configurable number of
    logistic classifiers. One of the classifiers in the mixture is not trained,
    and always predicts 0.

## Frame-Level Models

*   `LstmModel`: Processes the features for each frame using a multi-layered
    LSTM neural net. The final internal state of the LSTM is input to a
    video-level model for classification. Note that you will need to change the
    learning rate to 0.001 when using this model.
*   `DbofModel`: Projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then uses a
    video-level model to classify the now aggregated features.
*   `FrameLevelLogisticModel`: Equivalent to 'LogisticModel', but performs
    average-pooling on the fly over frame-level features rather than using
    pre-aggregated features.
