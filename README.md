# SC1015 Mini-Project - FireWatch 🔥👀

Nanyang Technological University \
School of Computer Science and Engineering \
Class: FCS1 \
Team: 2

## Overview 🧐

Welcome to FireWatch, a SC1015 Mini Project dedicated to the detection of fires through image analysis. By harnessing the power of cutting-edge computer vision and machine learning technologies, FireWatch is designed to swiftly identify the presence of fire in images captured by cameras or sensors. Our mission is to provide a reliable and efficient solution for early fire detection, ultimately contributing to enhanced safety and protection against wildfires.

## Objectives 🎯

- Employ advanced image processing techniques to enhance image quality and extract pertinent features.
- Implement state-of-the-art machine learning algorithms to classify images based on the presence of fire.
- Evaluate and optimize the performance of the fire detection model to ensure reliability and efficiency.

---
### Table of Contents:
1. Problem Formulation
2. Data Preparation and Cleaning
3. Exploratory Data Analysis
4. Random Forest Classifier
5. Insights & Conclusion
6. References
---
### 1. Problem Formulation
**Dataset Used:** [FIRE Dataset on Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset) \
**Question:** Would it be possible to achieve detection of fire via images? 🤔

**Success:** Determined using blobs formed by CV2 \
**Exceptions:** Outliers/Anomalies found in blobbing images with intense lighting or color mixtures closely resembling brown 🚨🔍

**Rationale:** Our team firmly believes that both the dataset and question we are exploring hold significant relevance in the global context. As students of SCSE, the development of a fire detection model holds immense potential in providing early warnings to individuals within the vicinity, ultimately facilitating prompt action to isolate and mitigate fires. 


### 2. Data Preparation and Cleaning

In this phase, we've prepared and processed our dataset to enhance our data analysis and efficiently use our data for the machine learning portion. Since our data comprises images, we've developed blob2.py to aid in generating the dataset.

Here's what we've accomplished:

Development of blob2.py: This custom script assists in pinpointing areas within the images where fire could be detected by utilizing blobbing techniques, thereby capturing color and intensity values and storing them in a CSV format.

Preliminary Feature Selection: Out of the 16 available variables, we've carefully chosen `15` relevant ones. Among these, `14` serve as predictors, while the remaining variable, `fire` serves as the response.

Identification of Zero Rows: During our analysis, we observed several rows containing all zeros, indicating that blobbing was not performed on those corresponding images. To validate this, we cross-checked the image names against these zero rows, ensuring that our Python code didn't overlook blobbing for them, affirming that they represent non-fire images.


### 3. Exploratory Data Analysis

In this section, we delved into the data to streamline the predictors used in our machine learning model, aiming to reduce complexity and computation time.

Our approach involved:

<b>Visualization with catplot:</b> We initially used a catplot to visualize the distribution of fire and non-fire images in our dataset. This revealed a significant class imbalance, with approximately 70% more fire images than non-fire images. To address this, we undertook a rebalancing strategy, undersampling the fire images to mitigate potential overfitting.

<b>Exploring data spread:</b> We examined the data spread both before and after rebalancing, focusing on variables related to color intensity. Notably, the spread of these variables appeared similar. We identified `std_h` and `blob_count` as the most skewed variables suitable for transformation.

<b>Transformation selection:</b> Opting for the Yeo-Johnson Transformation, we chose a method capable of handling zero values and eliminating the need for specifying an initial transformation parameter, λ.

<b>Q-Q Plot analysis:</b> Q-Q Plots were utilized to visualize the effect of transformation on the data distribution. A substantial shift in skewness was observed for both `std_h` and `blob_count`, indicating a closer resemblance to a normal distribution post-transformation.

By executing these steps, we aimed to refine our dataset and prepare it for improved model performance.

### 4. Data Driven Insights

To conduct further analysis, we plotted all variables using box plots to identify potential predictors for the response variable (fire).

1. Examining outliers and data spread, we observed clear distinctions between fire and non-fire images for variables such as `avg_r`, `avg_h`, `avg_s`, `avg_v`, `std_s`, `std_v`. These variables exhibit notable differences in data spread between fire and non-fire images, indicating potential relationships worthy of further investigation.

2. Employing point-biserial correlation, we evaluated the correlation between predictors and the response variable. Notably, `std_r`, `std_s`, and `std_v` displayed the highest positive correlations, suggesting they tend to have higher values when fire is present. Despite `size` also exhibiting a high positive correlation, it has numerous outliers, as depicted in the box plot. Furthermore, the p-values associated with `std_r`, `std_s`, and `std_v` were smaller than 0.05, indicating a significant relationship beyond random chance.

3. Consequently, we identified `std_r`, `std_s`, and `std_v` as the primary predictors for the response variable, based on their strong correlations and statistical significance.

Through this process, we refined our selection of predictors, focusing on those most likely to contribute meaningfully to our model's predictive power.

### 5. Classification

After deliberation, we opted to employ a random forest classifier due to its effectiveness in handling non-linear relationships and interactions among a large number of features.

1. To optimize the model's performance, we conducted experiments varying the number of trees and depth parameters.

2. Through experimentation, we determined that setting the number of trees to around 25 and the depth to 4 produced a well-fitted model. This configuration struck a balance, ensuring the model's complexity was adequate without risking overfitting on the training set, which could lead to significant disparities between training and test set results.

3. Subsequently, we trained the model and evaluated its goodness of fit. The accuracy of the model was found to be quite satisfactory.



## Contributors 👩‍💻👨‍💻

- [Anthony See](https://github.com/slightly-unrelated)
- [Tay Yuan Geng](https://github.com/TayYuanGeng)
- [Nicholas Tan](https://github.com/nichtyq)

Let's make the world a safer place with FireWatch! 🌍🔥🚀
