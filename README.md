# Case Study in Healthcare Using Pyspark Machine Learning and Logistical Regression to Extract Comorbidities / Co-diagnoses of Leukemia

## Problem Statement
There exists a normalized set of codes used in electronic health records (EHR) for the sake of describing a patient’s ailments during a hospital visit. This is known as the International Coding Definitions (ICD) for diagnoses. ICD is the standard for electronically coded medical data. However, diagnosis of diseases and medical conditions are rarely one note. Many diagnoses and diseases are commonly found alongside others. These are called codiagnoses and comorbidities respectively. The goal of this project is to use machine learning on a dataset of EHR to predict a subset of potential codiagnoses for any specified disease. For demonstration, we will choose 'leukemia' as the target disease.

## High Level Overview of Steps
1. Install required software
2. Download and import patient data as DataFrame (diagnoses_icd_demo.csv) into Spark
3. Import table with ICD code definitions (d_icd_diagnoses.csv)
4. Extract ICD codes that refer to specific diagnosis/disease using df.filter() method, in this demo I
select ICD codes for leukemia (be sure to query for other disease identifiers such as “leukemic”)
5. Filter patient DataFrame to only contain rows that match ICD codes
6. Extract list of hospital stays that involve leukemia (a hospital ID that matches extracted ICD
codes)
7. Go through patient data and tag all hospital stays involving leukemia (true/false or 1/0)
8. Prep patient data for machine learning pipeline (feature engineering, vectorization)
9. Create train and test splits of patient dataframe
10. Train a logistic regression model on the training data set
11. Evaluate the model (area under Receiver Operating Characteristics curve, accuracy, etc.)
12. Make predictions on the test set using the trained model
13. Plot most common co-diagnoses
14. Extract resulting set of codiagnoses in data format of choice (for simplicity, I exported as .csv)

## Dataset
MIMIC-IV: a free, publicly available “relational database containing real hospital stays for patients admitted to a tertiary academic medical center in Boston, MA” [(MIMIC-IV Documentation)](https://mimic-iv.mit.edu/docs/). MIMIC-IV is the fourth iteration of the MIMIC medical health data repository of over 40,000 patients admitted to the Beth Israel Deaconess Medical Center (BIDMC). It was completely de-identified as of MIMIC-III with the removal of patient identifiers in accordance with the Health Insurance Portability and Accountability Act (HIPAA).

- Full dataset size: 6.9 GB (.zip download from PhysioNet repository) - **REQUIRES CREDENTIALING**
- Demo dataset: d_icd_diagnoses.csv (8.8 MB), diagnoses_icd_demo.csv (27MB)
- Link to the demo data: https://bit.ly/3fi6y9L (since GitHub allows a maximum size of 25MB)

### Disclaimer:

The data provided in the data/ directory is a modified version of the original, which was previously downloaded from MIMIC-IV's data repository. This was done because this data set is semi-restricted. This means that while this dataset is open to all researchers, one must first go through a credentialing process via PhysioNet and signing a user agreement prior to being granted access.

For instructions on how to gain access to the data: https://mimic-iv.mit.edu/docs/access/

The credentialing exists to educate the researcher on responsible data handling of patient health data and Health Insurance Portability and Accountability Act (HIPAA) regulations.

All `subject_id` and `hadm_id` have been offset by some number to further preserve the anonymity of the patient data.

## Hardware Environment
Macbook Pro (2013) - Dual-Core i7 (3 Ghz) | 8 GB 1600 HHz DDR3 | Intel HD Graphics 4000 MacOS X Catalina (10.15.7)

## Software
- Python (3.8.5) - Anaconda Distribution: https://www.anaconda.com/products/individual 
- Jupyter Notebook (6.1.4) - Packaged with Anaconda
- PySpark (3.0.1) - https://spark.apache.org/downloads.html

## Lessons Learned (Pros/Cons)
- MIMIC-IV is a clean, highly organized, and well-documented dataset so researchers can hit the floor running. 
- My code demo implementation is not very efficient or optimized. 
- The accuracy of the model is highly dependent on a large dataset. 
- Larger patient datasets contain more recorded occurrences of varied diagnoses while a smaller dataset may lack in this regard. 
- A key limitation is the fact that the resulting co-diagnoses are representative only of the patient population of that specific medical center. For a more robust representation, the dataset needs to be bigger and include different populations.

## References
MIMIC-IV Dataset and Documentation:

https://mimic-iv.mit.edu https://mimic-iv.mit.edu/docs

Pyspark BinaryClassifcation Evaluator:

https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.BinaryClassificationEvaluator.html

Overview of Leukemia and Common Treatments:

https://www.mayoclinic.org/diseases-conditions/leukemia/diagnosis-treatment/drc-20374378
