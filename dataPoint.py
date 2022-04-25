import numpy
from sklearn.preprocessing import StandardScaler

class dataPoint:

  def __init__(self, x, y, ph, tds, turbidity):
    newScaler = StandardScaler(with_mean=True, with_std=True)
    newScaler.mean_ = numpy.asarray([7.08109924e+00, 2.20144658e+04, 3.96703817e+00])
    newScaler.var_ = numpy.asarray([2.16092001e+00, 7.68873781e+07, 6.08643197e-01])
    newScaler.scale_ = numpy.asarray([1.47000681e+00, 8.76854481e+03, 7.80155880e-01])
    # rangeArray = [14.0, 60906.259999999995, 5.29]
    self.loc = numpy.asarray([x,y])
    self.values = [ph, tds, turbidity]
    self.values = newScaler.transform(numpy.asarray(self.values).reshape(1,-1))[0]

  def __str__(self):
    return f'x coord: {self.loc[0]}, y coord: {self.loc[1]}'
  
  def __repr__(self):
    return f'({self.loc[0]}, {self.loc[1]})'