# SGMI-FS
Scalable Global Mutual Information-based Feature Selection Framework (SGMI).
=============================================================================
This is a new feature selection framework to cope with large-scale datasets in the distributed environment. The SGMI framework generates the similarity matrix in a distributed and scalable way and in a single pass. Then, it applies an optimization method, such as QP, SR, TP, to find a global solution in order to make a ranking of features. 
* QP => Quadratic Programming
* SR => Spectral Relaxation
* TP => Truncate Power


## Example (ml):
    import org.apache.spark.ml.feature._
    val method = "SR"
    val model = new FeatureSelector()
     .setOptMethod(method)
     .setMaxBin(10)
     .setBatchSize(0.25)
     .setFeaturesCol("features")
     .setLabelCol("label")
     .setUseCatch(true)
     .fit(discTrain)
     
## Important notes:
* The SGMI framework needs a discretized dataset. To this matter, applying the simple QuantileDiscretizer or Minimum Description Length Discretizer (MDLP) method proposed by Ramírez‐Gallego Sergio would be proper.
 * The maximum number of unique values in features should be set as MaxBin property.  
 * When the given dataset has substantial dimensions(as an example, more than 500), apply a lower number than 1.0, such as 0.25.
 * SR optimization method has better or comparable results than two other methods, QP and TP, but it is more time-consuming.
