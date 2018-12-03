class Car():
    def __init__(self, car_id):
        self.car_id = car_id
        self.group = 0
        self.mode = ('gas', 'ele')
        self.battery_level = float('inf')
        self.origin = None
        self.destination = None

    def set_battery_level(self, val):
        self.battery_level = val

    def set_origin(self, org):
        self.origin = org

    def set_destination(self, des):
        self.destination = des

    def get_od_pair(self):
        return (self.origin, self.destination)

    def get_battery_level(self):
        return self.battery_level

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self = self)

    def __str__(self):
        return '{self.__class__.__name__}: car_id = {self.car_id}'.format(self = self)