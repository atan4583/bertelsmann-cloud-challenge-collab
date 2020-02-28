# Udacity Bertelsmann Technical Scholarship Cloud Track Challenge Project - Deploy An AI Sentiment Prediction App to AWS Cloud
This repo is the project page of the 3 project repos and contains information about the project.
 ---
#### _The project was created by 3 scholars from the Cloud Track Challenge_

![png](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-projpage/tree/master/assets/BertelsmannChallenge.png)
## Project Website: :star2: **[AI Sentiment Prediction App on AWS](http://ai-frontend.s3-website-us-west-2.amazonaws.com/)** :star2:
#### For cost saving, only **_1_** instance of AI prediction engine is up for demo, though scaling up to **_3_**  instances is feasible

## Presentation: view this page or [Google Drive (PowerPoint)](https://drive.google.com/file/d/1nfianScb00vW7owKBzpQHafxhsjp4IdJ/view?usp=sharing) or [Github (PDF)](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-projpage/blob/master/AIAppOnAWS.pdf)

## AI Sentiment Prediction App on AWS infrastructure
![png](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-projpage/tree/master/assets/RNNappOnAWS.png)

## Project Artifact Repositories
![png](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-projpage/tree/master/assets/cicdworkflow.png)
The project has 3 code and artifact repositories:
### [ai-frontend](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-frontend)
> * this repo contains the project website static files **_index.html_** and **_app.js_**
>
> * the files reside in the **_static_** folder, any changes pushed onto the master branch will trigger the GitHub CI/CD Action on the repo to copy the static files to the S3 bucket hosting the project website on AWS

### [ai-automation](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-automation)
> * this repo contains the Serverless Framework configuration file **_serverless.yml_** and Lambda function code files for deployment of Lambda functions, their triggering events and required infrastructure resources (DynamoDB, API Gateway and S3) to AWS
>
> * any changes pushed to the master branch will trigger the Github CI/CD Action on the repo to start serverless deployment of the changes to AWS
>
>
### [ai-backend](https://github.com/bertelsmann-cloud-challenge-collaborate/ai-backend)
> * this repo contains the code files for building and pushing a Flask docker image to ECR, then deploying a new task definition to ECS
>
> * any changes pushed to the master branch will trigger the Github CI/CD Action on the repo to apply and deploy the changes to AWS
