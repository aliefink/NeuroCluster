---
title: 'NeuroCluster: A Python toolbox for nonparametric cluster-based statistical testing of neurophysiological data with respect to continuous predictors.'
tags:
  - Python
  - neurophysiology
  - non-parametric statistics
  - cluster-based permutation testing
  - spectro-temporal resolution
  - human intracranial electrophysiology
  - complex behavioral predictors
authors:
  - name: Alexandra Fink Skular
    orcid: 0000-0003-1648-4604
    equal-contrib: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Christina Maher
    orcid: 0009-0003-8188-2083
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2, 3"
  - name: Salman Qasim
    orcid: 0000-0001-8739-5962
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: "2, 3"
  - name: Ignacio Saezfth
    orcid: 0000-0003-0651-2069
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1, 2, 3, 4, 5"
affiliations:
 - name: The Nash Family Department of Neuroscience, Icahn School of Medicine at Mount Sinai, NY, NY
   index: 1
 - name: The Nash Family Center for Advanced Circuit Therapeutics, The Mount Sinai Hopsital, NY, NY
   index: 2
 - name: The Center for Computational Psychiatry,Icahn School of Medicine at Mount Sinai, NY, NY
   index: 3
- name: Department of Neurosurgery, The Mount Sinai Hopsital, NY, NY 
   index: 4
- name: Department of Neurology, The Mount Sinai Hopsital, NY, NY 
   index: 5
date: 03 September 2024
bibliography: paper.bib

---

# Summary

Cognitive neurophysiology offers a novel framework for studying cognitive brain-behavior relationships by relating electrophysiological signals to complex behaviors. With the advent of new biotechnologies and neurosurgical practices, large-scale human (and animal) intracranial electrophysiological recordings are becoming widely accessible. As a result, cognitive neurophysiologists can design cognitive experiments that leverage both the spatiotemporal resolution of electrophysiological data and the complexity of continuous behavioral variables (example citations). Analyzing these data requires sophisticated statistical methods that can interpret multidimensional neurophysiological data and dynamic, continuous behavioral variables. Classical statistical frameworks for analyzing event-related time series data are ill-equipped to manage the high dimensionality and behavioral complexity of cognitive neurophysiology studies. NeuroCluster is an open-source Python toolbox for analysis of multivariate electrophysiological data related to complex, continuous behavioral variables. NeuroCluster introduces a novel statistical approach, which uses non-parametric cluster-based permutation testing to identify time-frequency clusters of oscillatory power modulations that significantly encode time-varying, continuous behavioral variables. It also supports multivariate analyses by allowing for multiple behavioral predictors to model neural activity. NeuroCluster addresses a methodological gap in statistical approaches to relate continuous, cognitive predictors to underlying electrophysiological activity with time and frequency resolution, to determine the neurocomputational processes giving rise to complex behaviors. 

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# NeuroCluster

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Documentation

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
