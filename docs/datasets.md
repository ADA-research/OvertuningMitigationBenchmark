# Datasets
| #  | Dataset ID | Name                                       | # Instances | # Features | Type       |
| --:| ---------- | ------------------------------------------ | ----------: | ---------: | ---------- |
| 1  | 363612     | airfoil_self_noise                         |       1,503 |          5 | Regression |
| 2  | 363613     | Amazon_employee_access                     |      32,769 |          9 | Binary     |
| 3  | 363614     | anneal                                     |         898 |         38 | Multiclass |
| 4  | 363615     | Another-Dataset-on-used-Fiat-500           |       1,538 |          7 | Regression |
| 5  | 363616     | APSFailure                                 |      76,000 |        170 | Binary     |
| 6  | 363618     | bank-marketing                             |      45,211 |         13 | Binary     |
| 7  | 363619     | Bank_Customer_Churn                        |      10,000 |         10 | Binary     |
| 8  | 363620     | Bioresponse                                |       3,751 |      1,776 | Binary     |
| 9  | 363621     | blood-transfusion-service-center           |         748 |          4 | Binary     |
| 10 | 363623     | churn                                      |       5,000 |         19 | Binary     |
| 11 | 363624     | coil2000_insurance_policies                |       9,822 |         85 | Binary     |
| 12 | 363625     | concrete_compressive_strength              |       1,030 |          8 | Regression |
| 13 | 363626     | credit-g                                   |       1,000 |         20 | Binary     |
| 14 | 363627     | credit_card_clients_default                |      30,000 |         23 | Binary     |
| 15 | 363628     | customer_satisfaction_in_airline           |     129,880 |         21 | Binary     |
| 16 | 363629     | diabetes                                   |         768 |          8 | Binary     |
| 17 | 363630     | Diabetes130US                              |      71,518 |         47 | Binary     |
| 18 | 363631     | diamonds                                   |      53,940 |          9 | Regression |
| 19 | 363632     | E-CommereShippingData                      |      10,999 |         10 | Binary     |
| 20 | 363671     | Fitness_Club                               |       1,500 |          6 | Binary     |
| 21 | 363672     | Food_Delivery_Time                         |      45,451 |          9 | Regression |
| 22 | 363673     | GiveMeSomeCredit                           |     150,000 |         10 | Binary     |
| 23 | 363674     | hazelnut-spread-contaminant-detection      |       2,400 |         30 | Binary     |
| 24 | 363675     | healthcare_insurance_expenses              |       1,338 |          6 | Regression |
| 25 | 363676     | heloc                                      |      10,459 |         23 | Binary     |
| 26 | 363677     | hiva_agnostic                              |       3,845 |      1,617 | Multiclass |
| 27 | 363678     | houses                                     |      20,640 |          8 | Regression |
| 28 | 363679     | HR_Analytics_Job_Change_of_Data_Scientists |      19,158 |         12 | Binary     |
| 29 | 363681     | in_vehicle_coupon_recommendation           |      12,684 |         24 | Binary     |
| 30 | 363682     | Is-this-a-good-customer                    |       1,723 |         13 | Binary     |
| 31 | 363683     | kddcup09_appetency                         |      50,000 |        212 | Binary     |
| 32 | 363684     | Marketing_Campaign                         |       2,240 |         25 | Binary     |
| 33 | 363685     | maternal_health_risk                       |       1,014 |          6 | Multiclass |
| 34 | 363686     | miami_housing                              |      13,776 |         15 | Regression |
| 35 | 363689     | NATICUSdroid                               |       7,491 |         86 | Binary     |
| 36 | 363691     | online_shoppers_intention                  |      12,330 |         17 | Binary     |
| 37 | 363693     | physiochemical_protein                     |      45,730 |          9 | Regression |
| 38 | 363694     | polish_companies_bankruptcy                |       5,910 |         64 | Binary     |
| 39 | 363696     | qsar-biodeg                                |       1,054 |         41 | Binary     |
| 40 | 363697     | QSAR-TID-11                                |       5,742 |      1,024 | Regression |
| 41 | 363698     | QSAR_fish_toxicity                         |         907 |          6 | Regression |
| 42 | 363699     | SDSS17                                     |      78,053 |         11 | Multiclass |
| 43 | 363700     | seismic-bumps                              |       2,584 |         15 | Binary     |
| 44 | 363702     | splice                                     |       3,190 |         60 | Multiclass |
| 45 | 363704     | students_dropout_and_academic_success      |       4,424 |         36 | Multiclass |
| 46 | 363705     | superconductivity                          |      21,263 |         81 | Regression |
| 47 | 363706     | taiwanese_bankruptcy_prediction            |       6,819 |         94 | Binary     |
| 48 | 363707     | website_phishing                           |       1,353 |          9 | Multiclass |
| 49 | 363708     | wine_quality                               |       6,497 |         12 | Regression |
| 50 | 363711     | MIC                                        |       1,699 |        111 | Multiclass |
| 51 | 363712     | jm1                                        |      10,885 |         21 | Binary     |


Too big for RealMLP to finish within 60 hours: 
363628 (129k samples)
363673 (150k samples)
363679 (19k samples? Maybe problematic with one-hot?)
363681 (12k samples, some runs)
363684 (around 60 hours, but probably fine)

THINGS TO PICK UP TOMORROW: 
- Analyze the three RealMLP runs, make decision on actual experiments
- Make decision on logloss for multiclass
- Still have to fix big datasets, but then at least we are getting close
- Continue writing
- Continue analyzing results
- Formalize trajectories calculation
- - Probably one trajectory file per dataset per way of selecting incumbent