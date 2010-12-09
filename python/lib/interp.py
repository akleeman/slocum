class NcdfInterpolator():

    def __init__(self, filename, axesnames, variablename):
        from Scientific.IO.NetCDF import NetCDFFile
        self.file = NetCDFFile(filename, 'r')
        self.axes = map(lambda n, f=self.file: f.variables[n], axesnames)
        self.var = self.file.variables[variablename]
        for axis in self.axes:
            self.shape = self.shape + axis.shape

    def __call__(self, *points):
        """
        @returns: the function value obtained by linear interpolation
        @rtype: number
        @raise TypeError: if the number of arguments (C{len(points)})
            does not match the number of variables of the function
        @raise ValueError: if the evaluation point is outside of the
            domain of definition and no default value is defined
        """
        if len(points) != len(self.axes):
            raise TypeError('Wrong number of arguments')
        if len(points) == 1:
            # Fast Pyrex implementation for the important special case
            # of a function of one variable with all arrays of type double.
            period = self.period[0]
            if period is None: period = 0.
            try:
                return _interpolate(points[0], self.axes[0],
                                    self.values, period)
            except:
                # Run the Python version if anything goes wrong
                pass
        import pdb; pdb.set_trace()
        try:
            neighbours = map(_lookup, points, self.axes, self.period)
        except ValueError, text:
            if self.default is not None:
                return self.default
            else:
                raise ValueError(text)
        slices = sum([item[0] for item in neighbours], ())
        values = self.values[slices]
        for item in neighbours:
            weight = item[1]
        import pdb; pdb.set_trace()
            values = (1.-weight)*values[0]+weight*values[1]
        return values