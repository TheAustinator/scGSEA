class SinglePopulationError(ValueError):
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return f"This comparative analysis expected multiple populations, but got only 1: {self.label}"
