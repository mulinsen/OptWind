# OptWind
Basic wind farm design optimization tool in pure Python. The current version can apply to offshore wind farms or 
onshore wind farms at flat terrain, in which the wind resource can be non-uniformly distributed, but wake effects are calculated assuming
no terrain effects. The wind resource can be imported from WAsP grd files, i.e., those of sector wise Weibull-A, Weibull-k and frequency.
The main methodology is documented in two publications: 

Feng, J., & Shen, W. Z. (2015). Solving the wind farm layout optimization problem using random search algorithm. Renewable Energy, 78, 182-192. https://doi.org/10.1016/j.renene.2015.01.005

Feng, J., & Shen, W. Z. (2015). Modelling wind for wind farm layout optimization using joint distribution of wind speed and wind direction. Energies, 8(4), 3075-3092. https://doi.org/10.3390/en8043075
