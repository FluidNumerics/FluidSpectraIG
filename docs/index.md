# Normal Mode Analysis Spectra



## Motivation

### Friends don't let friends use Fourier analysis
For logically rectangular domains with periodic boundary conditions, Fourier Transforms are well suited to calculate spectra. However, velocity fields in regional ocean/atmosphere simulations are not periodic and they usually are defined on domains that have holes and are not logically rectangular. The mismatch of the properties of Fourier basis functions and regional ocean data necessitates workarounds that can influence spectra calculations. Normal mode analysis, however, provides a method to create basis functions that fit your model's geometry and ascribe energy to specific length scales, without modifying your model's data.


### A screw is a nail if all you have is a hammer
With Fourier Analysis, the basis functions are complex exponentials. In 2-D, the basis functions are obtained by multiplying by complex exponentials in $x$ and $y$ directions. When projecting non-periodic data onto periodic basis functions, the spectral slopes decay slowly (less than $\mathcal{O}(k^{1})$). This causes complications for researchers who wish to measure spectral slopes. Clever researchers have developed methods to hide these issues through prerequisite data manipulation.

To handle the lack of periodicity, windowing is usually performed wherein the velocity fields are multiplied by a function that decays to zero smoothly near the domain boundaries. 

Additionally, 2-D planes of data formed by intersecting regional model domains with a plane at constant depth result in "islands" or "holes". To handle holes, researchers usually interpolate through these discontinuities, introducing data where there was not data before. 

!!! note ""
    <center>**These workarounds arose simply because we had not considered an alternative approach that is designed to handle irregular geometries and non-periodicity.**</center>



### Really look at what we're doing
The goal of spectral analysis is to explain the distribution of energy across a range of length scales. This is done by projecting data onto a set of basis functions that are provably associated with specific length scales. Fourier basis functions are the eigenmodes of Laplace's equation with periodic boundary conditions. We can generalize this idea, by using appropriate Sturm-Liouville boundary value problems to generate basis functions that are more suitable for the data we wish to understand.

The Normal Mode Analysis (NMA) uses a Helmholz decomposition of a 2-D vector field, alongside the eigenmodes of the Laplacian operator with homogeneous Dirichlet boundary conditions to define the basis functions for 2-D velocity fields. This provides a spectra that can be attributed to rotational and divergent motions at distinct length scales. As a corollary to this, the NMA methodology provides Parseval's equality, meaning that all energy is accounted for through the NMA decomposition. 

Closed form solutions for the eigenmodes are not possible, generally. Because of this, the eigenmodes and eigenvalues are computed numerically using the Implicitly Resarted Arnoldi Method. With this approach, a subset of the spectra is calculated and a residual is tracked. The spectra can be focused on the largest length scales, smallest length scales, or targeted around a desired length scale using various strategies with IRAM.
