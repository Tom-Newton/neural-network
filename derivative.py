class Derivative:
    def __init__(self):
        self.components = []
    
    def __add__(self, new_component):
        self.components.append(new_component)
        return self

    def get_complete_derivative(self):
        components = self.components
        def complete_derivative(Y):
            evaluated_components = [component(Y) for component in components]
            return sum(evaluated_components)
        return complete_derivative