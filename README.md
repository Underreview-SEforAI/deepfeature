# deepfeature: Feature map testing for Deep Neural Networks.

## How to reproduce the experiment results of DeepFeature?


### First, train a DNN model.
```
python train.py
```

### Second, select test cases from the unlabel dataset.

```
python testcase_selection.py
```

### Next, calculate the FDR and FDD.

```
oython fault_detection_diversity.py
```

### Then evaluate the fuzzing procedure.

```
jupyter notebook -> fuzzing.ipynb.
```
