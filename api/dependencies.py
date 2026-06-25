from api.services import model_loader


def get_model():
    return model_loader.get_model()


def get_preprocessor():
    return model_loader.get_preprocessor()
