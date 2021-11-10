import numpy

from models.transformthem import TransformThem
from models.vse import VSE


def get_model(config):
    if 'vse' in config['model']['name']:
        model = VSE(config)
    elif config['model']['name'] == 'transformit':
        model = TransformIt(config)
    elif config['model']['name'] == 'transformthem':
        model = TransformThem(config)

    return model


def dot_sim(x, y):
    return numpy.dot(x, y.T)


def cosine_sim(x, y):
    x = x / numpy.expand_dims(numpy.linalg.norm(x, axis=1), 1)
    y = y / numpy.expand_dims(numpy.linalg.norm(y, axis=1), 1)
    return numpy.dot(x, y.T)
