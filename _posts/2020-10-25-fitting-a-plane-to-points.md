---
sitemap:
  lastmod: 2020-10-25
  priority: 0.6
  changefreq: 'monthly'
  exclude: 'no'
layout: post
title:  "Fitting a Plane to Points in Python"
subtitle: "A computationally efficient approach which remains robust even with noisy inputs"
date:   2020-10-24 23:04 -0400
tags: Python, NumPy, Numba
categories:
  - blog
unlisted: false
---
<br />
<hr />
<br />

<h3><strong>Introduction:</strong></h3>
While researching geometric methods for some private code, I stumbled upon a blogpost titled [**"Fitting a plane to noisy points in 3D"**](http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html) by Emil Ernerfeldt.
<br />
<br />

This post is an extension of his previous article, [**"Fitting a plane to many points in 3D"**](http://www.ilikebigbits.com/2015_03_04_plane_from_points.html), and together they provide an incredible explanation of how to efficiently compute a best-fit plane for points in three-dimensional space.
<br />
<br />

The culmination of Emil's work is a 81-line program (which appears to be written in Rust, though I'm not certain) which can take a set of 3-vectors (**X**, **Y**, **Z**) as input and return the centroid point along with the normal associated with a best-fit plane for those points. His code can be found at the end of the first post linked at the beginning of this article.
<br />
<br />

Below, I have written a Python version of the same algorithm. Specifically, my implementation uses the **Numba** library to achieve *just-in-time compilation* for maximum computational efficiency. This allows the code to be used within Python programs, while achieving an execution speed much closer to that of C/C++. The compilation is handled entirely by Numba and controlled with a single function decorator. To disable this Numba functionality for compatibility reasons, simply comment-out the `@jit(nopython=True)` line at the beginning of the function.
<br />
<br />

<h3><strong>Differences in my Implementation:</strong></h3>
<ul>
    <li>The results are extended to compute A, B, C, and D coefficients representing the plane</li>
    <li>Normalization is performed to ensure that (A, B, C) is a vector of length 1.0</li>
</ul>
**NOTE:** The coefficients returned follow the convention **Ax + By + Cz + D = 0**, with all variables on the same side of the equation.
<br />
<br />

<h3><strong>The Code:</strong></h3>
<h5>(Also available as a <a href="https://gist.github.com/ryanbranch/8aa3f0768c6cb9268296468d63f8f21c">GitHub Gist</a>)</h5>
{% highlight python %}
import numpy
from numba import jit

# Calculates the A, B, C, D coefficients of a normalized plane which best fits a dataset
# Based on an approach by Emil Ernerfeldt, titled "Fitting a plane to noisy points in 3D"
#   (http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html)
# Written by Ryan Branch on 2020/10/25. See https://ryanbran.ch/blog/2020/10/25/fitting-a-plane-to-points.html
# INPUTS: A 2D Numpy float array of arbitrary outer length, and inner length 3 for (x, y, z) of N points
# OUTPUTS: A 4-tuple of floats A, B, C, and D such that (Ax + By + Cz + D == 0)
#            (Returns None if no single valid solution exists)
# NOTE: The single line of code below enables Numba just-in-time compilation; comment it out to disable
@jit(nopython=True)
def computeBestFitPlane(points):
    valType = numpy.float64
    
    n = points.shape[0]  # n: the integer number of (X, Y, Z) coordinate triples in points
    if n < 3:
        return None
    
    # Determination of (X, Y, Z) coordinates of centroid ("average" point along each axis in dataset)
    sum = numpy.zeros((3), dtype=valType)
    for p in points:
        sum += p
    centroid = sum * (1.0 / valType(n))
    
    # Uses Emil Ernerfeldt's technique to calculate the full 3x3 covariance matrix, excluding symmetries
    xx = 0.0
    xy = 0.0
    xz = 0.0
    yy = 0.0
    yz = 0.0
    zz = 0.0
    for p in points:
        r = p - centroid
        xx += r[0] * r[0]
        xy += r[0] * r[1]
        xz += r[0] * r[2]
        yy += r[1] * r[1]
        yz += r[1] * r[2]
        zz += r[2] * r[2]
    xx /= valType(n)
    xy /= valType(n)
    xz /= valType(n)
    yy /= valType(n)
    yz /= valType(n)
    zz /= valType(n)
    
    weighted_dir = numpy.zeros((3), dtype=valType)
    axis_dir = numpy.zeros((3), dtype=valType)
    
    # X COMPONENT
    det_x = (yy * zz) - (yz * yz)
    axis_dir[0] = det_x
    axis_dir[1] = (xz * yz) - (xy * zz)
    axis_dir[2] = (xy * yz) - (xz * yy)
    weight = det_x * det_x
    if numpy.dot(weighted_dir, axis_dir) < 0.0:
        weight *= -1.0
    weighted_dir += axis_dir * weight
    
    # Y COMPONENT
    det_y = (xx * zz) - (xz * xz)
    axis_dir[0] = (xz * yz) - (xy * zz)
    axis_dir[1] = det_y
    axis_dir[2] = (xy * xz) - (yz * xx)
    weight = det_y * det_y
    if numpy.dot(weighted_dir, axis_dir) < 0.0:
        weight *= -1.0
    weighted_dir += axis_dir * weight
    
    # Z COMPONENT
    det_z = (xx * yy) - (xy * xy)
    axis_dir[0] = (xy * yz) - (xz * yy)
    axis_dir[1] = (xy * xz) - (yz * xx)
    axis_dir[2] = det_z
    weight = det_z * det_z
    if numpy.dot(weighted_dir, axis_dir) < 0.0:
        weight *= -1.0
    weighted_dir += axis_dir * weight
    
    a = weighted_dir[0]
    b = weighted_dir[1]
    c = weighted_dir[2]
    d = numpy.dot(weighted_dir, centroid) * -1.0  # Multiplication by -1 preserves the sign (+) of D on the LHS
    normalizationFactor = math.sqrt((a * a) + (b * b) + (c * c))
    if normalizationFactor == 0:
        return None
    elif normalizationFactor != 1.0:  # Skips normalization if already normalized
        a /= normalizationFactor
        b /= normalizationFactor
        c /= normalizationFactor
        d /= normalizationFactor
    # Returns a float 4-tuple of the A/B/C/D coefficients such that (Ax + By + Cz + D == 0)
    return (a, b, c, d)

{% endhighlight %}
