MODEL_LIST = {}

def register_model(name):

    def register_model_cls(cls):
        print(cls)
        MODEL_LIST[name] = cls
        return cls

    return register_model_cls


def build_model(parser,name):
    if name in MODEL_LIST:
        MODEL_LIST[name].add_args(parser)
        return MODEL_LIST[name]
    else:
        raise ValueError(f"{name} is not in MODEL_LIST.")

from .multitask_tagger import MultiTaskTagger