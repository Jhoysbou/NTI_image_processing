"""
Objects for detection
"""


class Object:
    # position – tuple of pixels!!!
    # x and y coordinates (e.g. (x,y) )
    #
    # color - String
    # Can be 'RED', 'YELLOW', 'GREEN', and 'BLUE'
    def __init__(self, position, color):
        self.position = position
        self.color = color

    def get_position(self):
        return self.position

    def get_color(self):
        return self.color

    def __str__(self):
        return '{} at position {} with color {}'.format(self.__class__.__name__, self.position, self.color)


class Cube(Object):
    def __init__(self, position, color):
        super().__init__(position, color)


class Bucket(Object):
    # radius - int, pixels!
    def __init__(self, position, color, radius):
        super().__init__(position, color)
        self.radius = radius

    def get_radius(self):
        return self.radius
