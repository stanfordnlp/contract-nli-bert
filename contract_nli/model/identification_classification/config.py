def update_config(config, *, impossible_strategy, class_loss_weight):
    class IdentificationClassificationConfig(type(config)):
        def __init__(self, impossible_strategy='ignore',
                     class_loss_weight=1.0, **kwargs):
            super().__init__(**kwargs)
            self.impossible_strategy = impossible_strategy
            self.class_loss_weight = class_loss_weight

        @classmethod
        def from_config(cls, config, *, impossible_strategy,
                        class_loss_weight):
            kwargs = config.to_dict()
            assert 'impossible_strategy' not in kwargs
            kwargs['impossible_strategy'] = impossible_strategy
            assert 'class_loss_weight' not in kwargs
            kwargs['class_loss_weight'] = class_loss_weight
            return cls(**kwargs)

    return IdentificationClassificationConfig.from_config(
        config, impossible_strategy=impossible_strategy,
        class_loss_weight=class_loss_weight)
