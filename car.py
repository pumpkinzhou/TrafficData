class Car():
    def __init__(self, car_id):
        self.car_id = car_id
        self.group_id = 0
        self.mode = 'gas'
        self.origin = None
        self.destination = None

    def set_origin(self, org):
        self.origin = org

    def set_destination(self, des):
        self.destination = des

    def get_od_pair(self):
        return (self.origin, self.destination)

    def __repr__(self):
        return '{self.__class__.__name__}'.format(self = self)

    def __str__(self):
        return '{self.__class__.__name__}: car_id = {self.car_id}'.format(self = self)


class ECar(Car):
    def __init__(self, car_id):
        Car.__init__(self, car_id)
        self.group_id = 1
        self.battery_level = float('inf')
        self.mode = 'ele'

    def set_battery_level(self, val):
        self.battery_level = val

    def get_battery_level(self):
        return self.battery_level


class HCar(ECar):
    def __init__(self, car_id):
        ECar.__init__(self, car_id)
        self.group_id = 2
        self.mode =  {0: 'gas', 1: 'ele'}


if __name__ == "__main__":
    h_car = HCar('1')
    print(h_car)

