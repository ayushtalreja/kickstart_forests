
# Review

* `requirements.txt` contains many unnecessary dependencies.

* The dataset drops N/A values, which is not necessarily admissible.

* The dataset unnecessarily drops the tree species feature. (Why?)

* The dataset sampling option is unnecessary. (YAGNI)

* None of the columns `w***` is used to train the models. Why discard them? 
  There is a code part in the `DataSet` class, which addresses these.
  However, 
  * this would be the wrong place to do it.
  * the method TSNE is not suitable as dimensionality reduction technique
    within a model, because it does not learn a mapping which could be applied
    to new data.

* `hyperopt` script is unfinished; was not applied.

* The model factory contains some redundant models. 
  Why have the `orig` and `pure` variants in addition to the ones that use a feature 
  collector? Note that in the refactoring journey, the simpler, earlier models 
  exist only such that we can compare them to newer models. 
  The feature collector-based model is functionally equivalent.

  The `v2` (`pure`) models could have been handled by making the factory methods
  parametrisable.

* Normalisation rule for sentinel levels features was incompletely specified.

* The MLP model would have benefitted from scaling/normalisation and target
  transformation. Why did you choose to not use the normalisation?

* Set of models in main app is oddly controlled via two Boolean flags.