from hypopredict.daniel_model.lstmcnn import Lstmcnnmodel


def build_model(config, n_features):
    if config["architecture"]["type"] == "lstm_cnn":
        return Lstmcnnmodel(config, n_features)
    elif config["architecture"]["type"] == "resnet1d":
        return Lstmcnnmodel(config, n_features)
    else:
        raise ValueError("Unknown architecture")
