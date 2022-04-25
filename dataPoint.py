import numpy


class dataPoint:

  def __init__(self, x, y, ph, tds, turbidity):
    # rangeArray = [14.0, 60906.259999999995, 5.29]
    self.loc = numpy.asarray([x,y])
    self.values = (ph, tds, turbidity)

  def __str__(self):
    return f'x coord: {self.loc[0]}, y coord: {self.loc[1]}'
  
  def __repr__(self):
    return f'({self.loc[0]}, {self.loc[1]})'