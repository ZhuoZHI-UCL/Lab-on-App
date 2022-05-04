# Lab-on-App: AI Empowered Point-of-Care Diagnostics for Ageing Population
_The repository for Zhuo ZHI's Ph.D project_

<img src="https://github.com/ZhuoZHI-UCL/Lab-on-App/blob/main/Introduction/image/Smartphone-app-to-assess-anemia-1.jpg"/>

This repository contains the information of Zhuo ZHI’s PhD project (Dept Electronic and
Electrical Engineering, University College London). The content of this README.md document
is as follows:

1. Introduction of the project
2. Structure of the repository
3. Participants of the project

##Introduction of the project
###1. Background of the anemia

Anaemia is a serious global public health problem that particularly affects young children and
pregnant women. WHO estimates that 42% of children less than 5 years of age and 40% of
pregnant women and 20% of the older Population worldwide are anaemic.

Most commonly, people with anemia report feelings of weakness or fatigue, and sometimes
poor concentration. They may also report shortness of breath on exertion. In very severe anemia,
The patient may have symptoms related to this, such as palpitations, angina (if pre-existing heart
disease is present), intermittent claudication of the legs, and symptoms of heart failure.

Therefore, the diagnosis of anemia and its causes is important for improving human well-being.

###2. Problems existing in traditional anemia detection methods

Anaemia, defined as reduced haemoglobin concentration. The diagnosis of anemia in men is based
on a hemoglobin of less than 130 to 140 g/L (13 to 14 g/dL). In women, it is less than 120 to 130
g/L (12 to 13 g/dL). The traditional diagnosis of anaemia requires laboratory-based measurements
of a venous blood sample, which could bring trauma and pain to patients, even wound infection.
The diagnosis process involves complex equipment which requires professional operators and fixed
test site. To address these issues, researchers keep looking for novel anemia detection methods.

###3. Problems existing in novel anemia detection methods

Novel anemia detection methods based on AI data analysis and advanced sensing technology. For
the first part, the ML/DL models can be built for classifying patients’ finger nail, fundu, conjuctival
or other bio-images into anemia and normal condition. However, these models are not able to
give the cause of the anemia or predict the anemia in a early stage. At the same time, the
privacy protection agreement and time/labor cost make it difficult to collect biometric images of
patients at scale, which limits the performance of models. For the second part, researchers have
developed sensors based on the principle of multispectral analysis, transmission spectrum analysis,
etc. Similarly, these methods can not explain the reason of the anemia and it is unable to iterate
and update the algorithm with new biomarks.

###4. What do we want to achieve?

We would like to develop a Lab-on-App to non-invasively diagnose anaemia and its causes (e.g.
genetics, diet, or injury) that can be easily used by older people, carers, or healthcare professionals.
It is a system with portable device and corresponding software deployment. It is anticipated that
the successful demonstration of our proposed Lab-on-App will lead to additional work by this team
using mobile health technology to diagnose other conditions afflicting older population (kidney
diseases, colon diseases, or vitamin deficiencies).

###5. How do we achieve the goal?

The project solution consists of five parts: the patient data acquisition, the data analysis, the
diagnostic result, the hardware integration and the software deployment. The project solution is
shown as follows.

<img src="https://github.com/ZhuoZHI-UCL/Lab-on-App/blob/main/Introduction/image/image.png"/>

1. The patient data acquisition (The patient data are collected through four parts.)
    - EHR data
   
      EHR data contain the demographic, vitals, lab test results, disease and medication
      records of each patient. We will get the access to it and select useful information.
   
    - Non-invasive biometric sensor
   
      We will develop the non-invasive biometric sensor based on multispectral (or other prin-
      ciples) to measure the hemoglobin, blood oxygen concentration and other biometrics
      from skin.
2. The data analysis

   We will develop the multi-modal model to analysis different kinds of data and combine the
   useful information for regression or classification. The model involves the function of prepro-
   cessing, imputation, feature selection, interpretation, uncertainty and calibration, etc.
3. The diagnostic result

   The diagnostic result consists of three parts.
   - The diagnosis of anemia as well as its causes.
   - The suggested personalized treatment planning, for example, the medication and some
mechanical equipments.
   - The diagnosis of other diseases.
4. The hardware integration
    
   We would like to integrate all sensors, MCU, battery, etc into a compact, portable and low-
   power platform. The integrated module design, the wireless/wired communication and the
   embedded system development are the main parts that need attention.
5. The software deployment (The software deployment includes three parts.)
   - App development
   
     An IOS/ Android app will be developed and all functions are performed in it.
   - Cloud computing and data interaction
   
     The ML/DL model will be deployed in the server due to the limited computing ability of
     the mobile device. All patient data are also stored on the server. Distributed computing
     and privacy Computing are needed to be considered for the process.
   - Database development
   
     A database needs to be established for all patients to facilitate version iteration and
     update
###6. The innovation of our solution
There are two main innovation of our solution.
1. EHR data is combined for the diagnosis
   - Multiple health histories of users are modeled to analyze and predict disease occurrence
and reasonable suggestions can be given.
   - Combining the EHR data with other biomarks have shown better performance than
single-input model.
   - The continuous update of EHR data can realize the monitoring and prediction of the
patient’s condition.
2. Truly portable and intelligent diagnosis
   - Truly portable diagnosis
   
     No external power supply is required. The system has low power consumption, Low
     weight and size.
   - The cloud service
     EHR and sensor data can be uploaded to the cloud to maximize user history.
   - Intelligent diagnosis
     Machine Learning models are built for diagnosis. The disease prediction, diagnosis and
     health advice can be given.
   - User universality
     No professional operation is needed, which is suitable for user with all age.
##Structure of the repository
The structure of the repository is shown as follows.
<img src="https://github.com/ZhuoZHI-UCL/Lab-on-App/blob/main/Introduction/image/structure%20of%20the%20github%20project.jpg"/>

The repository consists of three parts: the progress record, the action points record and the
code.
1. The progress record
    
   The progress record records weekly progress and presents it in .pdf, .ppt, etc.
2. The action points record

   The action points record records weekly action points by weekly meeting and presents it in
.docx.
3. The code

   It is the most important part of the project. We firstly divide the code into four parts
(EHR data, image data, other biomarks data and multimodal data) according to the data
type. Then, for each kind of data, we create separate folder for data from different sources (eg.
EHR data from Stanford University (EHR-SU)). Finally, three steps are performed on selected
data: the data preprocessing, ML/DL model building and evaluation and analysis. The data
preprocessing includes outlier detection, normalization, feature selection, imputation, data
clipping, etc. The structure and weights of the ML/DL model will be stored in ML/DL
model building. In evaluation and analysis step, we will propose the required evaluation
metrics and conduct comparative experiments as well as analyse the result.
##Participants of the project
1. Project Supervisor
   - Professor Miguel Rodrigues, Dept Electronic and Electrical Engineering, University Col-
lege London
2. Project Co-Supervisors
   - Dr Mine Orlu, UCL School of Pharmacy, University College London
   - Professor Andreas Demosthenous, Dept Electronic and Electrical Engineering, Univer-
sity College London
3. Project Collaborators
   - Dr Moe Elbadawi, UCL School of Pharmacy, University College London
   - Professor Abdul Basit, UCL School of Pharmacy, University College London
   - Dr Adam Daneshmend, Imperial College Healthcare NHS Trust






