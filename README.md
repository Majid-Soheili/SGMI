# SGMI-FS
Scalable Global Mutual Information based Feature Selection Framework
This is a proposed feature selection framework to cope with big datasets
such that three optimizing methods are applied to rank features.
These optimizing methods are, QP, SR, TP.
QP => Quadratic Programming
SR => Spectral Relaxation
TP => Truncate Power

In addition to mentioned optimization methods, each arbitrary optimization method that be proper to solve BQ (Binary Quadratic) problem can be applied.
The experimental study illustrates that the execution time of the produced method strongly depends on making similarity matrix, so the execution time of