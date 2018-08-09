#Coordinate Systems
## Geographic coordinate system
This is most populat coordinate system representing point in lat, long, elevation

## Projected coordinate system
This system assume earth as flat 2D surface. This is useful when doing measurements in meters. Only problem with this system here is that number of digit in x,y are 6 to 7. This makes very difficult for certain algorithms to work on. So it is necessary to get relative coordinates

## ECEF coordinate system
This earth-centered, earth-fixed system consider earth center of mass as fixed origin point.

### Reference lla (lat,long,altitude)
This point is taken as reference while running sfm algorithm

### Topocentric reference
The topocentric reference frame is a metric one with the origin
at the given (lat, lon, alt) position, with the X axis heading east,
the Y axis heading north and the Z axis vertical to the ellipsoid.

Here reference lla is taken as topocentric reference in ECEF coordinate system not in geographic coordinates.

```
topocentric_from_lla(lat, lon, alt, reflat, reflon, refalt):
```
will convert lat,long,alt point to relative to ref(lat,long,alt) in ECEF coordinate system. 

To get back lat,long,alt from topocentric reference system use the following. Here x,y,z are relative location of point wrt to reference point

```
lla_from_topocentric(x,y,z,reflat,reflong,refalt)
```
