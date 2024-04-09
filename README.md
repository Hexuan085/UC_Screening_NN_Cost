# UC_SCREENING_ML
The codes for 'Enabling Fast Unit Commitment Constraint Screening via Learning Cost Model'.

## For the sample-agnostic screening

1. Using 'XXbus_UC.py' to generate UC samples;
2. Using 'Cost_fitting_XXBUS.py' to train the neural network model;
3. Using 'XXbus_UC_limits.py' to conduct the standard optimization-based screening and the proposed ccost-driven screening;
4. Using 'Reduced problem.py' to compare the solution time of the two screening methods.

## For the sample-aware screening

1. Using 'XXbus_UC.py' to generate UC samples;
2. Using 'Cost_fitting_XXBUS.py' to train the neural network model;
3. Using 'XXbus_UC_limits.py' to conduct the standard optimization-based screening and the proposed ccost-driven screening, and then compare the average number of reduced constraints by the two screening methods.

## For KNN screening

1. Using 'Data generation.py' to get training data for KNN;
2. Using 'Reduced problem.py' to conduct KNN screening;
2. Using 'Reduced problem.py' to compare the results of KNN screening, the standard optimization-based screening and the proposed ccost-driven screening.

##
_If you do not plan to generate data by your own and want to see the process along with the results of constraint screening, please skip step 1-2 and run the corresponding code directly since
the required data already exists in the corresponding path._ 
